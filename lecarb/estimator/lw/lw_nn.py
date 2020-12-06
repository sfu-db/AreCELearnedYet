import time
import logging
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import LWNNModel
from .common import load_lw_dataset, encode_query, decode_label
from ..postgres import Postgres
from ..estimator import Estimator
from ..utils import report_model, evaluate, run_test
from ...dataset.dataset import load_table
from ...workload.workload import Query
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS

L = logging.getLogger(__name__)

class Args:
    def __init__(self, **kwargs):
        self.bs = 32
        self.epochs = 500
        self.lr = 0.001 # default value in both pytorch and keras
        self.hid_units = '128_64_32'
        self.bins = 200
        self.train_num = 10000

        # overwrite parameters from user
        self.__dict__.update(kwargs)

class LWQueryDataset(Dataset):
    def __init__(self, X, y, gt):
        super(LWQueryDataset, self).__init__()
        self.X = X
        self.y = y
        self.gt = gt
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.gt[idx]

def make_dataset(dataset, num=-1):
    X, y, gt = dataset
    L.info(f"{X.shape}, {y.shape}, {gt.shape}")
    if num <= 0:
        return LWQueryDataset(X, y, gt)
    else:
        return LWQueryDataset(X[:num], y[:num], gt[:num])

def train_lw_nn(seed, dataset, version, workload, params, sizelimit):
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # convert parameter dict of lw(nn)
    L.info(f"params: {params}")
    args = Args(**params)

    table = load_table(dataset, version)

    # create model
    fea_num = table.col_num*2+3
    model = LWNNModel(fea_num, args.hid_units).to(DEVICE)
    model_size = report_model(model)

    # check size limit
    if sizelimit > 0 and model_size > (sizelimit * table.data_size_mb):
        L.info(f"Exceeds size limit {model_size:.2f}MB > {sizelimit} x {table.data_size_mb}, do not conintue training!")
        return
    L.info(f'Overall LWNN model size = {model_size:.2f}MB')

    # load dataset
    dataset = load_lw_dataset(table, workload, seed, args.bins)
    train_dataset = make_dataset(dataset['train'], num=args.train_num)
    valid_dataset = make_dataset(dataset['valid'], num=args.train_num//10)

    L.info(f"Number of training samples: {len(train_dataset)}")
    L.info(f"Number of validation samples: {len(valid_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.bs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs)

    # Train model
    state = {
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': table.dataset,
        'version': table.version,
        'workload': workload,
        'model_size': model_size,
        'fea_num': fea_num,
    }
    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}_{workload}-{model.name()}_bin{args.bins}_ep{args.epochs}_bs{args.bs}_{args.train_num//1000}k-{seed}.pt"

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss(reduction='none')
    best_valid_loss = float('inf')

    start_stmp = time.time()
    valid_time = 0
    for epoch in range(args.epochs):
        train_loss = torch.tensor([])
        model.train()
        for _, data in enumerate(train_loader):
            inputs, labels, _ = data
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()

            optimizer.zero_grad()
            preds = model(inputs).reshape(-1)

            loss = mse_loss(preds, labels)
            loss.mean().backward()
            optimizer.step()
            train_loss = torch.cat([train_loss, loss.cpu()])
        dur_min = (time.time() - start_stmp) / 60
        L.info(f"Epoch {epoch+1}, loss: {train_loss.mean()}, time since start: {dur_min:.1f} mins")

        L.info(f"Test on valid set...")
        valid_stmp = time.time()
        valid_loss = torch.tensor([])
        valid_preds = torch.tensor([])
        valid_gts = torch.tensor([])
        model.eval()
        for _, data in enumerate(valid_loader):
            inputs, labels, gts = data
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()

            with torch.no_grad():
                preds = model(inputs).reshape(-1)
                valid_preds = torch.cat([valid_preds, preds.cpu()])
                valid_gts = torch.cat([valid_gts, gts.float()])

                loss = mse_loss(preds, labels)
                valid_loss = torch.cat([valid_loss, loss.cpu()])

        valid_loss = valid_loss.mean()
        L.info(f'Valid loss is {valid_loss:.4f}')
        valid_preds = np.maximum(np.round(decode_label(valid_preds)), 0.0)
        L.info("Q-Error on validation set:")
        _, metrics = evaluate(valid_preds, valid_gts)

        if valid_loss < best_valid_loss:
            L.info('best valid loss for now!')
            best_valid_loss = valid_loss
            state['model_state_dict'] = model.state_dict()
            state['optimizer_state_dict'] = optimizer.state_dict()
            state['valid_error'] = {workload: metrics}
            state['train_time'] = (valid_stmp-start_stmp-valid_time) / 60
            state['current_epoch'] = epoch
            torch.save(state, model_file)

        valid_time += time.time() - valid_stmp

    L.info(f"Training finished! Time spent since start: {(time.time()-start_stmp)/60:.2f} mins")
    L.info(f"Model saved to {model_file}, best valid: {state['valid_error']}")

class LWNN(Estimator):
    def __init__(self, model, model_name, pg_est, table):
        super(LWNN, self).__init__(table=table, model=model_name)
        self.model = model.to(DEVICE)
        self.model.eval()
        self.pg_est = pg_est

    def query(self, query):
        if isinstance(query, Query):
            query = encode_query(self.table, query, self.pg_est)
        return self.query_vector(query)

    def query_vector(self, vec):
        start_stmp = time.time()
        with torch.no_grad():
            pred = self.model(torch.FloatTensor(vec).to(DEVICE)).cpu().item()
        dur_ms = (time.time() - start_stmp) * 1e3
        return np.maximum(np.round(decode_label(pred)), 0.0), dur_ms

def load_lw_nn(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pt"
    L.info(f"load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']

    table = load_table(dataset, state['version'])
    # load model
    model = LWNNModel(state['fea_num'], args.hid_units).to(DEVICE)
    report_model(model)
    L.info(f"Overall LWNN model size = {state['model_size']:.2f}MB")
    model.load_state_dict(state['model_state_dict'])
    pg_est = Postgres(table, args.bins, state['seed'])

    estimator = LWNN(model, model_name, pg_est, table)
    return estimator, state

def test_lw_nn(dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        model: model file name
        use_cache: load processed vectors directly instead of build from queries
    """
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"Torch threads: {torch.get_num_threads()}")

    model_file = MODEL_ROOT / dataset / f"{params['model']}.pt"
    L.info(f"Load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']

    # load corresonding version of table
    table = load_table(dataset, state['version'])

    # load model
    model = LWNNModel(state['fea_num'], args.hid_units).to(DEVICE)
    report_model(model)
    L.info(f"Overall LWNN model size = {state['model_size']:.2f}MB")
    model.load_state_dict(state['model_state_dict'])

    if params['use_cache']:
        # do not need to connect postgres in this case
        estimator = LWNN(model, params['model'], None, table)
        L.info(f"Load and build lw(nn) estimator: {estimator}")

        # test table might has different version with train
        test_table = load_table(dataset, version)
        lw_dataset = load_lw_dataset(test_table, workload, state['seed'], args.bins)
        X, _, gt = lw_dataset['test']
        run_test(dataset, version, workload, estimator, overwrite, lw_vec=(X, gt))
    else:
        pg_est = Postgres(table, args.bins, state['seed'])
        estimator = LWNN(model, params['model'], pg_est, table)
        L.info(f"Load and build lw(nn) estimator: {estimator}")

        run_test(dataset, version, workload, estimator, overwrite)


