"""Model training."""
import time
import copy
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import made
from . import transformer
from ..estimator import Estimator, OPS
from ..utils import report_model, run_test, evaluate
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS, VALID_NUM_DATA_DRIVEN
from ...dataset.dataset import load_table
from ...workload.workload import load_queryset, load_labels, query_2_triple

L = logging.getLogger(__name__)

class Args:
    def __init__(self, **kwargs):
        # general parameters
        self.order = None
        self.num_orderings = 1
        self.bs = 2048
        self.epochs = 20
        self.column_masking = True
        self.constant_lr = None
        self.warmups = 0

        # Transformer TODO
        self.heads = 0
        self.blocks = 2
        self.dmodel = 32
        self.dff = 128
        self.transformer_act = 'gelu'

        # MADE & ResMADE
        self.fc_hiddens = 128
        self.layers = 4
        self.residual = True
        self.direct_io = True
        self.inv_order = False
        self.input_encoding = 'binary'
        self.output_encoding = 'one_hot'
        self.embed_size = 64
        self.embed_threshold = 128

        # overwrite parameters from user
        self.__dict__.update(kwargs)

class NaruTableDataset(Dataset):
    def __init__(self, table):
        super(NaruTableDataset, self).__init__()
        table = copy.deepcopy(table)
        self.tuples_np = np.stack([col.discretize(table.data[cname]) for cname, col in table.columns.items()], axis=1)
        self.tuples = torch.as_tensor(self.tuples_np.astype(np.float32, copy=False))

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        return self.tuples[idx]

def Entropy(name, data, bases=None):
    import scipy.stats
    s = 'Entropy of {}:'.format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == 'e' or base is None
        e = scipy.stats.entropy(data, base=base if base != 'e' else None)
        ret.append(e)
        unit = 'nats' if (base == 'e' or base is None) else 'bits'
        s += ' {:.4f} {}'.format(e, unit)
    L.info(s)
    return ret

def RunEpoch(args,
             split,
             model,
             opt,
             train_data,
             val_data=None,
             batch_size=100,
             upto=None,
             epoch_num=None,
             verbose=False,
             log_every=10,
             return_losses=False,
             table_bits=None):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    for step, xb in enumerate(loader):
        if split == 'train':
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                    .sum(-1).mean()
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))
                    loss = (-logps).mean()

        losses.append(loss.item())

        if (step+1) % log_every == 0:
            if split == 'train':
                L.info(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr'
                    .format(epoch_num+1, step+1, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, lr))
            else:
                L.info('{} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(split, step+1, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            L.info('%s epoch average loss: %f' % (split, np.mean(losses)))

    if return_losses:
        return losses
    return np.mean(losses)

def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def MakeMade(args, scale, cols_to_train, seed, fixed_ordering=None):
    if args.inv_order:
        L.info('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.vocab_size for c in cols_to_train]),
        input_bins=[c.vocab_size for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        #  embed_size=32,
        embed_size=args.embed_size,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        embed_threshold=args.embed_threshold,
        epoch=args.epochs
    ).to(DEVICE)

    return model


def MakeTransformer(args, cols_to_train, fixed_ordering, seed=None):
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.vocab_size for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)


def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)

def train_naru(seed, dataset, version, workload, params, sizelimit):
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    L.info(f"torch threads: {torch.get_num_threads()}")
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()

    torch.manual_seed(seed)
    np.random.seed(seed)

    table = load_table(dataset, version)

    # load validation queries and labels
    valid_queries = load_queryset(dataset, workload)['valid'][:VALID_NUM_DATA_DRIVEN]
    labels = load_labels(dataset, version, workload)['valid'][:VALID_NUM_DATA_DRIVEN]

    # convert parameter dict to original naru code format
    L.info(f"params: {params}")
    args = Args(**params)

    fixed_ordering = None
    if args.order is not None:
        L.info(f"Using passed-in order: {args.order}")
        fixed_ordering = args.order

    if args.heads > 0:
        model = MakeTransformer(args,
                                cols_to_train=list(table.columns.values()),
                                fixed_ordering=fixed_ordering,
                                seed=seed)
    else:
        model = MakeMade(
            args,
            scale=args.fc_hiddens,
            cols_to_train=list(table.columns.values()),
            seed=0, # force natrual_ordering=True
            fixed_ordering=fixed_ordering)

    mb = report_model(model)
    if sizelimit > 0 and mb > (sizelimit * table.data_size_mb):
        L.info(f"Exceeds size limit {mb:.2f}MB > {sizelimit} x {table.data_size_mb}, do not conintue training!")
        return

    if not isinstance(model, transformer.Transformer):
        L.info('Applying InitWeight()')
        model.apply(InitWeight)

    if isinstance(model, transformer.Transformer):
        opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        opt = torch.optim.Adam(list(model.parameters()), 2e-4)

    L.info(f"start building naru dataset for table {table.name}...")
    train_data = NaruTableDataset(table)
    L.info("dataset build finished")

    L.info('calculate table entropy...')
    df = pd.DataFrame(data=train_data.tuples_np)
    table_bits = Entropy(
        table.name,
        df.groupby(list(df.columns)).size(), [2])[0]

    train_start = time.time()
    for epoch in range(args.epochs):
        mean_epoch_train_loss = RunEpoch(
            args,
            'train',
            model,
            opt,
            train_data=train_data,
            val_data=train_data,
            batch_size=args.bs,
            epoch_num=epoch,
            log_every=200,
            table_bits=table_bits)

        dur_min = (time.time() - train_start) / 60
        L.info(f'epoch {epoch+1} train loss {mean_epoch_train_loss:.4f} nats / {mean_epoch_train_loss/np.log(2):.4f} bits, time since start: {dur_min:.1f} mins')

    dur_min = (time.time() - train_start) / 60
    L.info('Training finished! Time spent since start: {:.1f} mins'.format(dur_min))

    L.info('Evaluating likelihood on full data...')
    all_losses = RunEpoch(
        args,
        'test',
        model,
        train_data=train_data,
        val_data=train_data,
        opt=None,
        batch_size=args.bs,
        log_every=200,
        table_bits=table_bits,
        return_losses=True)
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    L.info(f"Evaluating on valid set with {VALID_NUM_DATA_DRIVEN} queries...")
    estimator = Naru(model,
                     'valid',
                     table,
                     2000, # hardcode 2000 psample for evaluation
                     device=DEVICE,
                     shortcircuit=args.column_masking)
    preds = []
    for q in valid_queries:
        est_card, _ = estimator.query(q)
        preds.append(est_card)
    _, metrics = evaluate(preds, [l.cardinality for l in labels])

    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_time': dur_min,
        'model_size': mb,
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': table.dataset,
        'version': table.version,
        'table_bits': table_bits,
        'model_bits': model_bits,
        'valid_error': {workload: metrics}
    }

    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-{model.name()}_warm{args.warmups}-{seed}.pt"
    torch.save(state, model_file)
    L.info(f'model saved to:{model_file}')

class Naru(Estimator):
    """Progressive sampling from Naru."""
    def __init__(
            self,
            model,
            model_name,
            table,
            r,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False  # Skip sampling on wildcards?
    ):
        super(Naru, self).__init__(table=table, model=model_name, psample=r)
        torch.set_grad_enabled(False)
        self.model = model
        self.shortcircuit = shortcircuit

        if r < 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.row_num

        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros(1, self.model.nin, device=device))

        self.dom_sizes = [c.vocab_size for c in self.table.columns.values()]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        # Inference optimizations below.

        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        if 'MADE' in str(model):
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    if layer.masked_weight is None:
                        layer.masked_weight = layer.mask * layer.weight
                        L.info('Setting masked_weight in MADE, do not retrain!')
        for p in model.parameters():
            p.detach_()
            p.requires_grad = False
        self.init_logits.detach_()

        with torch.no_grad():
            self.kZeros = torch.zeros(self.num_samples,
                                      self.model.nin,
                                      device=self.device)
            self.inp = self.traced_encode_input(self.kZeros)

            # For transformer, need to flatten [num cols, d_model].
            self.inp = self.inp.view(self.num_samples, -1)

    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  inp=None,
                  return_probs=False):
        ncols = len(columns)
        logits = self.init_logits
        if inp is None:
            inp = self.inp[:num_samples]
        masked_probs = []
        all_probs = [] # for analysis of naru's output

        # Use the query to filter each column's domain.
        valid_i_list = [None] * ncols  # None means all valid.
        for i in range(ncols):
            natural_idx = ordering[i]

            # Column i.
            col = self.table.columns[columns[natural_idx]]
            op = operators[natural_idx]
            if op is not None:
                # There exists a filter.
                valid_i = OPS[op](col.vocab,
                                  vals[natural_idx]).astype(np.float32,
                                                            copy=False)
                assert len(valid_i) == len(col.vocab), valid_i
                # Comparing with NaN will always be False
                assert not col.has_nan or not valid_i[0], col
            else:
                continue

            # This line triggers a host -> gpu copy, showing up as a
            # hotspot in cprofile.
            valid_i_list[i] = torch.as_tensor(valid_i, device=self.device)

        # Fill in wildcards, if enabled.
        if self.shortcircuit:
            for i in range(ncols):
                natural_idx = i if ordering is None else ordering[i]
                if operators[natural_idx] is None and natural_idx != ncols - 1:
                    if natural_idx == 0:
                        self.model.EncodeInput(
                            None,
                            natural_col=0,
                            out=inp[:, :self.model.
                                    input_bins_encoded_cumsum[0]])
                    else:
                        l = self.model.input_bins_encoded_cumsum[natural_idx -
                                                                 1]
                        r = self.model.input_bins_encoded_cumsum[natural_idx]
                        self.model.EncodeInput(None,
                                               natural_col=natural_idx,
                                               out=inp[:, l:r])

        # Actual progressive sampling.  Repeat:
        #   Sample next var from curr logits -> fill in next var
        #   Forward pass -> curr logits
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]

            if return_probs:
                all_probs.append(torch.softmax(self.model.logits_for_col(natural_idx, logits), 1))

            # If wildcard enabled, 'logits' wasn't assigned last iter.
            if not self.shortcircuit or operators[natural_idx] is not None:
                probs_i = torch.softmax(
                    self.model.logits_for_col(natural_idx, logits), 1)
                #  L.debug(f"{i},{probs_i.shape}, {probs_i}")

                valid_i = valid_i_list[i]
                if valid_i is not None:
                    probs_i *= valid_i

                probs_i_summed = probs_i.sum(1)

                masked_probs.append(probs_i_summed)

                # If some paths have vanished (~0 prob), assign some nonzero
                # mass to the whole row so that multinomial() doesn't complain.
                paths_vanished = (probs_i_summed <= 0).view(-1, 1)
                probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
                #  L.debug(f"{i}, {probs_i}")

            if i < ncols - 1:
                # Num samples to draw for column i.
                if i != 0:
                    num_i = 1
                else:
                    num_i = num_samples if num_samples else int(
                        self.r * self.dom_sizes[natural_idx])

                if self.shortcircuit and operators[natural_idx] is None:
                    data_to_encode = None
                else:
                    samples_i = torch.multinomial(
                        probs_i, num_samples=num_i,
                        replacement=True)  # [bs, num_i]
                    data_to_encode = samples_i.view(-1, 1)

                # Encode input: i.e., put sampled vars into input buffer.
                if data_to_encode is not None:  # Wildcards are encoded already.
                    if not isinstance(self.model, transformer.Transformer):
                        if natural_idx == 0:
                            self.model.EncodeInput(
                                data_to_encode,
                                natural_col=0,
                                out=inp[:, :self.model.
                                        input_bins_encoded_cumsum[0]])
                        else:
                            l = self.model.input_bins_encoded_cumsum[natural_idx
                                                                     - 1]
                            r = self.model.input_bins_encoded_cumsum[
                                natural_idx]
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])
                    else:
                        # Transformer.  Need special treatment due to
                        # right-shift.
                        l = (natural_idx + 1) * self.model.d_model
                        r = l + self.model.d_model
                        if i == 0:
                            # Let's also add E_pos=0 to SOS (if enabled).
                            # This is a no-op if disabled pos embs.
                            self.model.EncodeInput(
                                data_to_encode,  # Will ignore.
                                natural_col=-1,  # Signals SOS.
                                out=inp[:, :self.model.d_model])

                        if transformer.MASK_SCHEME == 1:
                            # Should encode natural_col \in [0, ncols).
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])
                        elif natural_idx < self.model.nin - 1:
                            # If scheme is 0, should not encode the last
                            # variable.
                            self.model.EncodeInput(data_to_encode,
                                                   natural_col=natural_idx,
                                                   out=inp[:, l:r])

                # Actual forward pass.
                next_natural_idx = i + 1 if ordering is None else ordering[i +
                                                                           1]
                if self.shortcircuit and operators[next_natural_idx] is None and return_probs is False:
                    # If next variable in line is wildcard, then don't do
                    # this forward pass.  Var 'logits' won't be accessed.
                    # But if we want to see the true probability predication for next column
                    # we have to run forward
                    continue

                if hasattr(self.model, 'do_forward'):
                    # With a specific ordering.
                    logits = self.model.do_forward(inp, ordering)
                else:
                    if self.traced_fwd is not None:
                        logits = self.traced_fwd(inp)
                    else:
                        logits = self.model.forward_with_encoded_input(inp)

        # deal with no predicates or one predicate
        #  print(masked_probs)
        if len(masked_probs) == 0:
            return 1, all_probs
        elif len(masked_probs) == 1:
            return masked_probs[0].mean().item(), all_probs
        # Doing this convoluted scheme because m_p[0] is a scalar, and
        # we want the corret shape to broadcast.
        p = masked_probs[1]
        for ls in masked_probs[2:]:
            p *= ls
        p *= masked_probs[0]

        return p.mean().item(), all_probs

    def query(self, query, return_probs=False):
        # Massages queries into natural order.
        columns, operators, vals = query_2_triple(query, with_none=True)

        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            L.info('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * len(columns)
        for natural_idx in range(len(columns)):
            inv_ordering[ordering[natural_idx]] = natural_idx

        with torch.no_grad():
            inp_buf = self.inp.zero_()
            # Fast (?) path.
            if num_orderings == 1:
                ordering = orderings[0]
                start_stmp = time.time()
                p, all_probs = self._sample_n(
                    self.num_samples,
                    ordering if isinstance(
                        self.model, transformer.Transformer) else inv_ordering,
                    columns,
                    operators,
                    vals,
                    inp=inp_buf,
                    return_probs=return_probs)
                dur_ms = (time.time() - start_stmp) * 1e3
                if return_probs:
                    return np.round(p * self.cardinality).astype(dtype=np.int32,
                                                                copy=False), dur_ms, all_probs
                return np.round(p * self.cardinality).astype(dtype=np.int32,
                                                            copy=False), dur_ms

            # Num orderings > 1.
            ps = []
            start_stmp = time.time()
            for ordering in orderings:
                p_scalar, all_probs = self._sample_n(self.num_samples // num_orderings,
                                          ordering, columns, operators, vals, return_probs=return_probs)
                ps.append(p_scalar)
            dur_ms = (time.time() - start_stmp) * 1e3
            if return_probs:
                return np.round(np.mean(ps) * self.cardinality).astype(
                    dtype=np.int32, copy=False), dur_ms, all_probs
            return np.round(np.mean(ps) * self.cardinality).astype(
                dtype=np.int32, copy=False), dur_ms

def load_naru(dataset: str, model_name: str, psample: int) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']
    if not hasattr(args, 'embed_threshold'):
        args.embed_threshold = 128

    # load corresonding version of table
    table = load_table(dataset, state['version'])

    if args.heads > 0:
        model = MakeTransformer(args,
                                cols_to_train=list(table.columns.values()),
                                fixed_ordering=args.order,
                                seed=args.seed)
    else:
        model = MakeMade(args,
                         scale=args.fc_hiddens,
                         cols_to_train=list(table.columns.values()),
                         seed=0,
                         fixed_ordering=args.order)
    report_model(model)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    estimator = Naru(model,
                     model_name,
                     table,
                     psample,
                     device=DEVICE,
                     shortcircuit=args.column_masking)

    L.info(f"load and built naru estimator: {estimator}")
    return estimator, state


def test_naru(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        model: model file name
        psample: number of progressive sample during each inference
    """
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    model_file = MODEL_ROOT / dataset / f"{params['model']}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']
    if not hasattr(args, 'embed_threshold'):
        args.embed_threshold = 128

    # load corresonding version of table
    table = load_table(dataset, state['version'])

    if args.heads > 0:
        model = MakeTransformer(args,
                                cols_to_train=list(table.columns.values()),
                                fixed_ordering=args.order,
                                seed=args.seed)
    else:
        model = MakeMade(args,
                         scale=args.fc_hiddens,
                         cols_to_train=list(table.columns.values()),
                         seed=0,
                         fixed_ordering=args.order)
    report_model(model)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    estimator = Naru(model,
                     params['model'],
                     table,
                     params['psample'],
                     device=DEVICE,
                     shortcircuit=args.column_masking)

    L.info(f"load and built naru estimator: {estimator}")

    # init random seed before progressive sampling
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_test(dataset, version, workload, estimator, overwrite)

def update_naru(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    model_file = MODEL_ROOT / dataset / f"{params['model']}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']
    new_args = Args(**params)
    epochs = 1
    if new_args.epochs:
        epochs = new_args.epochs
    if not hasattr(args, 'embed_threshold'):
        args.embed_threshold = 128
    
    # load validation queries and labels
    valid_queries = load_queryset(dataset, workload)['valid'][:VALID_NUM_DATA_DRIVEN]
    labels = load_labels(dataset, version, workload)['valid'][:VALID_NUM_DATA_DRIVEN]

    # load new version of table
    table = load_table(dataset, version)

    if args.heads > 0:
        model = MakeTransformer(args,
                                cols_to_train=list(table.columns.values()),
                                fixed_ordering=args.order,
                                seed=args.seed)
    else:
        model = MakeMade(args,
                         scale=args.fc_hiddens,
                         cols_to_train=list(table.columns.values()),
                         seed=0,
                         fixed_ordering=args.order)
    report_model(model)
    model.load_state_dict(state['model_state_dict'])
    L.info(f"start building naru dataset for table {table.name}...")
    train_data = NaruTableDataset(table)
    L.info("dataset build finished")

    if isinstance(model, transformer.Transformer):
        opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        opt = torch.optim.Adam(list(model.parameters()), 2e-4)
    opt.load_state_dict(state['optimizer_state_dict'])

    L.info('calculate table entropy...')
    df = pd.DataFrame(data=train_data.tuples_np)
    table_bits = Entropy(
        table.name,
        df.groupby(list(df.columns)).size(), [2])[0]
    
    update_start = time.time()
    for epoch in range(epochs):
        mean_epoch_train_loss = RunEpoch(
                args,
                'train',
                model,
                opt,
                train_data=train_data,
                val_data=train_data,
                batch_size=args.bs,
                epoch_num=epoch,
                log_every=200,
                table_bits=table_bits)
    dur_min = (time.time() - update_start) / 60
    L.info(f'Update train loss {mean_epoch_train_loss:.4f} nats / {mean_epoch_train_loss/np.log(2):.4f} bits, time since start: {dur_min:.4f} mins')
        
    L.info('Evaluating likelihood on full data...')
    all_losses = RunEpoch(
        args,
        'test',
        model,
        train_data=train_data,
        val_data=train_data,
        opt=None,
        batch_size=args.bs,
        log_every=200,
        table_bits=table_bits,
        return_losses=True)
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    L.info(f"Evaluating on valid set with {VALID_NUM_DATA_DRIVEN} queries...")
    estimator = Naru(model,
                     'valid',
                     table,
                     2000, # hardcode 2000 psample for evaluation
                     device=DEVICE,
                     shortcircuit=args.column_masking)
    preds = []
    for q in valid_queries:
        est_card, _ = estimator.query(q)
        preds.append(est_card)
    _, metrics = evaluate(preds, [l.cardinality for l in labels])

    new_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_time': state['train_time'],
        'model_size': state['model_size'],
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': table.dataset,
        'version': table.version,
        'table_bits': table_bits,
        'model_bits': model_bits,
        'valid_error': {workload: metrics},
        'update_time': dur_min
    }
    model.epoch = epochs
    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-{model.name()}_warm{args.warmups}-{seed}.pt"
    torch.save(new_state, model_file)
    L.info(f'model saved to:{model_file}')