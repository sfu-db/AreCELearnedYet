import time
import logging
from typing import Dict, Any, Tuple
import pickle

import numpy as np
import xgboost as xgb

from .common import load_lw_dataset, encode_query, decode_label
from ..postgres import Postgres
from ..estimator import Estimator
from ..utils import evaluate, run_test
from ...dataset.dataset import load_table
from ...workload.workload import Query
from ...constants import MODEL_ROOT, NUM_THREADS, PKL_PROTO

L = logging.getLogger(__name__)

class Args:
    def __init__(self, **kwargs):
        self.trees = 16
        self.bins = 200
        self.train_num = 10000

        # overwrite parameters from user
        self.__dict__.update(kwargs)

def train_lw_tree(seed, dataset, version, workload, params, sizelimit):
    np.random.seed(seed)

    # convert parameter dict of lw(nn)
    L.info(f"params: {params}")
    args = Args(**params)
    valid_num = args.train_num // 10

    table = load_table(dataset, version)
    dataset = load_lw_dataset(table, workload, seed, args.bins)
    train_X, train_y, _ = dataset['train']
    valid_X, valid_y, valid_gt = dataset['valid']

    # Train model
    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}_{workload}-lwxgb_tr{args.trees}_bin{args.bins}_{args.train_num//1000}k-{seed}.pkl"

    L.info(f"Start training...")
    start_stmp = time.time()
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=args.trees, random_state=seed, n_jobs=NUM_THREADS)
    model.fit(train_X[:args.train_num], train_y[:args.train_num], eval_set=[(valid_X[:valid_num], valid_y[:valid_num])])
    dur_min = (time.time() - start_stmp) / 60
    L.info(f"Finish training, time since start: {dur_min:.4f} mins")

    L.info(f"Run on valid set...")
    preds = np.maximum(np.round(decode_label(model.predict(valid_X[:valid_num]))), 0.0)
    gts = valid_gt[:valid_num]
    L.info("Q-Error on validation set:")
    _, metrics = evaluate(preds, gts)

    state = {
        'seed': seed,
        'args': args,
        'device': 'cpu',
        'threads': NUM_THREADS,
        'dataset': table.dataset,
        'version': table.version,
        'workload': workload,
        'model': model,
        'train_time': dur_min,
        'valid_error': {workload: metrics}
        #  'model_size': model_size,
    }
    with open(model_file, 'wb') as f:
        pickle.dump(state, f, protocol=PKL_PROTO)

    L.info(f'All finished! Time spent since training start: {(time.time()-start_stmp)/60:.2f} mins')
    L.info(f"Model saved to {model_file}")

class LWTree(Estimator):
    def __init__(self, model, model_name, pg_est, table):
        super(LWTree, self).__init__(table=table, model=model_name)
        self.model = model
        self.pg_est = pg_est

    def query(self, query):
        if isinstance(query, Query):
            query = encode_query(self.table, query, self.pg_est)
        return self.query_vector(np.expand_dims(query, axis=0))

    def query_vector(self, vec):
        start_stmp = time.time()
        pred = self.model.predict(vec).item()
        dur_ms = (time.time() - start_stmp) * 1e3
        return np.maximum(np.round(decode_label(pred)), 0.0), dur_ms


def load_lw_tree(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pkl"
    L.info(f"load model from {model_file} ...")
    with open(model_file, 'rb') as f:
        state = pickle.load(f)

    # load model
    args = state['args']
    model = state['model']
    table = load_table(dataset, state['version'])
    pg_est = Postgres(table, args.bins, state['seed'])

    estimator = LWTree(model, model_name, pg_est, table)
    return estimator, state

def test_lw_tree(dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        model: model file name
        use_cache: load processed vectors directly instead of build from queries
    """
    # uniform thread number
    model_file = MODEL_ROOT / dataset / f"{params['model']}.pkl"
    L.info(f"Load model from {model_file} ...")
    with open(model_file, 'rb') as f:
        state = pickle.load(f)

    # load corresonding version of table
    table = load_table(dataset, state['version'])

    # load model
    args = state['args']
    model = state['model']
    pg_est = Postgres(table, args.bins, state['seed'])
    estimator = LWTree(model, params['model'], pg_est, table)

    L.info(f"Load and built lw(tree) estimator: {estimator}")
    if params['use_cache']:
        # test table might has different version with train
        test_table = load_table(dataset, version)
        lw_dataset = load_lw_dataset(test_table, workload, state['seed'], args.bins)
        X, _, gt = lw_dataset['test']
        run_test(dataset, version, workload, estimator, overwrite, lw_vec=(X, gt))
    else:
        run_test(dataset, version, workload, estimator, overwrite)


