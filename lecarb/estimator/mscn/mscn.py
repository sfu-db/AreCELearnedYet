import time
import logging
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataset

from .model import SetConv
from ..estimator import Estimator, OPS
from ..utils import report_model, qerror, evaluate, run_test
from ...dataset.dataset import load_table
from ...workload.workload import load_queryset, load_labels, query_2_triple
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS

L = logging.getLogger(__name__)

class Args:
    def __init__(self, **kwargs):
        self.bs = 1024
        self.lr = 0.001
        self.epochs = 200
        self.num_samples = 1000
        self.hid_units = 256
        self.train_num = 100000

        # overwrite parameters from user
        self.__dict__.update(kwargs)

def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot

def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing

def normalize_labels(labels, min_val=None, max_val=None):
    # +1 to deal with 0 scenario
    labels = np.array([np.log(float(l.cardinality+1)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        L.info(f"min log(label): {min_val}")
    if max_val is None:
        max_val = labels.max()
        L.info(f"max log(label): {max_val}")
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val

def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    # -1 to deal with 0 scenario, need to restrict to >= 0
    return np.array(np.round(np.exp(labels) - 1), dtype=np.int64)

def get_sample_bitmap(sample, query):
    # do not need to convert [] to >= and <= here
    columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
    bitmap = np.ones(len(sample), dtype=bool)
    for c, o, v in zip(columns, operators, values):
        bitmap &= OPS[o](sample[c], v)
    return [bitmap.astype(int)]

def encode_data(table, query, column2vec, op2vec):
    columns, operators, values = query_2_triple(query, with_none=False, split_range=True)
    predicates_enc = []
    for c, o, v in zip(columns, operators, values):
        norm_val = table.columns[c].normalize(v)
        pred_vec = []
        pred_vec.append(column2vec[c])
        pred_vec.append(op2vec[o])
        pred_vec.append(norm_val)
        pred_vec = np.hstack(pred_vec)
        predicates_enc.append(pred_vec)
    # for no predicate scenario
    if len(predicates_enc) == 0:
        predicates_enc.append(np.zeros((len(column2vec) + len(op2vec) + 1)))
    return predicates_enc

def encode_datas(table, queries, column2vec, op2vec):
    return [encode_data(table, q, column2vec, op2vec) for q in queries]

def load_dicts(table):
    # Get column name dict
    # we assume any column can have predicates
    column2vec, _ = get_set_encoding(table.data.columns)

    # Get operator name dict
    # NOTICE: [] should be converted to two operators: >= and <= later for mscn
    operators = set(['=', '>=', '<='])
    op2vec, _ = get_set_encoding(operators)

    # Get min max value for each column
    return column2vec, op2vec

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals) - 1 # -1 since we +1 when normalize

def qerror_loss(preds, targets, min_val, max_val):
    errors = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        e = qerror(preds[i], targets[i])
        errors.append(e if torch.is_tensor(e) else torch.tensor([e], requires_grad=True, device=torch.device(DEVICE)))
    return torch.mean(torch.cat(errors))

def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, targets, sample_masks, predicate_masks = data_batch

        if cuda:
            samples, predicates, targets = samples.cuda(), predicates.cuda(), targets.cuda()
            sample_masks, predicate_masks = sample_masks.cuda(), predicate_masks.cuda()
        samples, predicates, targets = Variable(samples), Variable(predicates), Variable(
            targets)
        sample_masks, predicate_masks = Variable(sample_masks), Variable(predicate_masks)

        t = time.time()
        outputs = model(samples, predicates, sample_masks, predicate_masks)
        t_total += time.time() - t

        for i in range(outputs.shape[0]):
            preds.append(outputs[i].cpu().item())

    return preds, t_total

def make_sample_tensor_mask(sample):
    # no need to pad since only for single table
    sample_tensor = np.vstack(sample)
    sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
    return sample_tensor, sample_mask

def make_predicate_tensor_mask(predicate, max_pred):
    predicate_tensor = np.vstack(predicate)
    num_pad = max_pred - predicate_tensor.shape[0]
    predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
    predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
    predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
    return predicate_tensor, predicate_mask

def make_dataset(samples, predicates, labels, max_pred):
    """Add zero-padding and wrap as tensor dataset."""
    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor, sample_mask = make_sample_tensor_mask(sample)
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)
    L.debug(f'Sample tensor shape: {sample_tensors.shape}, mask shape: {sample_masks.shape}')

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor, predicate_mask = make_predicate_tensor_mask(predicate, max_pred)
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)
    L.debug(f'Predicate tensor shape: {predicate_tensors.shape}, mask shape: {predicate_masks.shape}')

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(sample_tensors, predicate_tensors, target_tensor,
                                 sample_masks, predicate_masks)

def train_mscn(seed, dataset, version, workload, params, sizelimit):
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # convert parameter dict of mscn
    L.info(f"params: {params}")
    args = Args(**params)

    table = load_table(dataset, version)
    L.info(f"Start loading queryset:{workload} and labels for version {version} of dataset {dataset}...")
    queryset = load_queryset(dataset, workload)
    labels = load_labels(dataset, version, workload)
    if args.train_num < len(queryset['train']):
        queryset['train'] = queryset['train'][:args.train_num]
        labels['train'] = labels['train'][:args.train_num]
    valid_num = args.train_num // 10
    if valid_num < len(queryset['valid']):
        queryset['valid'] = queryset['valid'][:valid_num]
        labels['valid'] = labels['valid'][:valid_num]
    L.info(f"Use {len(queryset['train'])} queries for train and {len(queryset['valid'])} queries for validation")

    # create model
    column2vec, op2vec = load_dicts(table)
    predicate_feats = len(column2vec) + len(op2vec) + 1
    model = SetConv(args.num_samples, predicate_feats, args.hid_units)
    model_size = report_model(model)

    # materialize sample
    sample = table.data.sample(n=args.num_samples, random_state=seed)
    sample_size = table.data_size_mb * (args.num_samples / table.row_num)

    # check size limit
    mscn_size = model_size + sample_size
    if sizelimit > 0 and mscn_size > (sizelimit * table.data_size_mb):
        L.info(f"Exceeds size limit {mscn_size:.2f}MB > {sizelimit} x {table.data_size_mb}, do not conintue training!")
        return
    L.info(f'Overall MSCN model size + sample size = {mscn_size:.2f}MB')

    # Get feature encoding and proper normalization
    samples_train = [get_sample_bitmap(sample, q) for q in queryset['train']]
    samples_valid = [get_sample_bitmap(sample, q) for q in queryset['valid']]
    predicates_train = [encode_data(table, q, column2vec, op2vec) for q in queryset['train']]
    predicates_valid = [encode_data(table, q, column2vec, op2vec) for q in queryset['valid']]
    label_norm, min_val, max_val = normalize_labels(labels['train'] + labels['valid'])
    labels_train = label_norm[:len(queryset['train'])]
    labels_valid = label_norm[len(queryset['train']):]
    L.info(f"Number of training samples: {len(labels_train)}")
    L.info(f"Number of validation samples: {len(labels_valid)}")

    # Train model
    # NOTICE: do not record min max value for each column, make sure to load the same table when test
    state = {
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': table.dataset,
        'version': table.version,
        'workload': workload,
        'model_size': mscn_size,
        'label_range': (min_val, max_val),
        'samples': sample
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_valid_loss = float('inf')
    cuda = False if DEVICE == 'cpu' else True

    if cuda:
        model.cuda()

    max_pred = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_valid]))
    train_dataset = make_dataset(samples_train, predicates_train,
                                 labels=labels_train, max_pred=max_pred)
    valid_dataset = make_dataset(samples_valid, predicates_valid,
                                 labels=labels_valid, max_pred=max_pred)
    train_data_loader = DataLoader(train_dataset, batch_size=args.bs)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.bs)

    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}_{workload}-{model.name()}_ep{args.epochs}_bs{args.bs}_{args.train_num//1000}k-{seed}.pt"

    model.train()
    L.info('start train mscn...')
    start_stmp = time.time()
    valid_time = 0
    for epoch in range(args.epochs):
        loss_total = 0.
        for batch_idx, data_batch in enumerate(train_data_loader):
            samples, predicates, targets, sample_masks, predicate_masks = data_batch
            if cuda:
                samples, predicates, targets = samples.cuda(), predicates.cuda(), targets.cuda()
                sample_masks, predicate_masks = sample_masks.cuda(), predicate_masks.cuda()
            samples, predicates, targets = Variable(samples), Variable(predicates), Variable(targets)
            sample_masks, predicate_masks = Variable(sample_masks), Variable(predicate_masks)
            optimizer.zero_grad()
            outputs = model(samples, predicates, sample_masks, predicate_masks)
            loss = qerror_loss(outputs, targets.float().reshape(-1, 1), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        dur_min = (time.time() - start_stmp) / 60
        L.info(f"Epoch {epoch+1}, loss: {loss_total/len(train_data_loader)}, time since start: {dur_min:.1f} mins")

        L.info(f"Test on valid set...")
        valid_stmp = time.time()
        preds_valid, _ = predict(model, valid_data_loader, cuda)

        # Unnormalize
        preds_valid_unnorm = unnormalize_labels(preds_valid, min_val, max_val)
        labels_valid_unnorm = unnormalize_labels(labels_valid, min_val, max_val)

        L.info("Q-Error on validation set:")
        _, metrics = evaluate(preds_valid_unnorm, labels_valid_unnorm)

        valid_loss = metrics['mean']
        if valid_loss < best_valid_loss:
            L.info(f'best valid loss for now: {valid_loss}(mean)!')
            best_valid_loss = valid_loss
            state['model_state_dict'] = model.state_dict()
            state['optimizer_state_dict'] = optimizer.state_dict()
            state['valid_error'] = {workload: metrics}
            state['train_time'] = (valid_stmp-start_stmp-valid_time) / 60
            state['current_epoch'] = epoch
            torch.save(state, model_file)

        valid_time += time.time() - valid_stmp

    L.info(f'Training finished! Time spent since start: {(time.time()-start_stmp)/60:.2f} mins')
    L.info(f"Model saved to {model_file}, best valid: {state['valid_error']}")

class MSCN(Estimator):
    def __init__(self, model, model_name, samples, table, column2vec, op2vec, label_range):
        super(MSCN, self).__init__(table=table, model=model_name)
        self.model = model
        self.samples = samples
        self.column2vec = column2vec
        self.op2vec = op2vec
        self.minval = label_range[0]
        self.maxval = label_range[1]
        self.device = torch.device(DEVICE)
        self.model.to(self.device)
        self.model.eval()

    def query(self, query):
        sample_enc = get_sample_bitmap(self.samples, query)
        predicate_enc = encode_data(self.table, query, self.column2vec, self.op2vec)

        sample_tensor, sample_mask = make_sample_tensor_mask(sample_enc)
        predicate_tensor, predicate_mask = make_predicate_tensor_mask(predicate_enc, len(predicate_enc))

        sample_tensor = torch.FloatTensor(np.expand_dims(sample_tensor, axis=0)).to(self.device)
        sample_mask = torch.FloatTensor(np.expand_dims(sample_mask, axis=0)).to(self.device)
        predicate_tensor = torch.FloatTensor(np.expand_dims(predicate_tensor, axis=0)).to(self.device)
        predicate_mask = torch.FloatTensor(np.expand_dims(predicate_mask, axis=0)).to(self.device)

        start_stmp = time.time()
        with torch.no_grad():
            pred = self.model(sample_tensor, predicate_tensor, sample_mask, predicate_mask)
        dur_ms = (time.time() - start_stmp) * 1e3

        return unnormalize_labels(pred.cpu(), self.minval, self.maxval).item(), dur_ms

def load_mscn(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']

    table = load_table(dataset, state['version'])
    # load model
    column2vec, op2vec = load_dicts(table)
    predicate_feats = len(column2vec) + len(op2vec) + 1
    model = SetConv(args.num_samples, predicate_feats, args.hid_units)
    report_model(model)
    L.info(f"Overall MSCN model size + sample size = {state['model_size']:.2f}MB")
    model.load_state_dict(state['model_state_dict'])

    estimator = MSCN(model,
                     model_name,
                     state['samples'],
                     table,
                     column2vec,
                     op2vec,
                     state['label_range'])

    return estimator, state


def test_mscn(dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        model: model file name
    """
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    model_file = MODEL_ROOT / dataset / f"{params['model']}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']

    # load corresonding version of table
    table = load_table(dataset, state['version'])

    # load model
    column2vec, op2vec = load_dicts(table)
    predicate_feats = len(column2vec) + len(op2vec) + 1
    model = SetConv(args.num_samples, predicate_feats, args.hid_units)
    report_model(model)
    L.info(f"Overall MSCN model size + sample size = {state['model_size']:.2f}MB")
    model.load_state_dict(state['model_state_dict'])

    estimator = MSCN(model,
                     params['model'],
                     state['samples'],
                     table,
                     column2vec,
                     op2vec,
                     state['label_range'])

    L.info(f"load and built mscn estimator: {estimator}")
    run_test(dataset, version, workload, estimator, overwrite)
