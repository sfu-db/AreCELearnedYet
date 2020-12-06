import csv
import ray
import logging
import numpy as np
import pandas as pd
import torch
from scipy.stats.mstats import gmean

#  from .lw.lw_nn import LWNN
#  from .lw.lw_tree import LWTree
from .estimator import Estimator
from ..constants import NUM_THREADS, RESULT_ROOT
from ..workload.workload import load_queryset, load_labels
from ..dataset.dataset import load_table

L = logging.getLogger(__name__)

def report_model(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    L.info(f'Number of model parameters: {num_params} (~= {mb:.2f}MB)')
    L.info(model)
    return mb

def qerror(est_card, card):
    if est_card == 0 and card == 0:
        return 1.0
    if est_card == 0:
        return card
    if card == 0:
        return est_card
    if est_card > card:
        return est_card / card
    else:
        return card / est_card

def rmserror(preds, labels, total_rows):
    return np.sqrt(np.mean(np.square(preds/total_rows-labels/total_rows)))

def evaluate(preds, labels, total_rows=-1):
    errors = []
    for i in range(len(preds)):
        errors.append(qerror(float(preds[i]), float(labels[i])))

    metrics = {
        'max': np.max(errors),
        '99th': np.percentile(errors, 99),
        '95th': np.percentile(errors, 95),
        '90th': np.percentile(errors, 90),
        'median': np.median(errors),
        'mean': np.mean(errors),
        'gmean': gmean(errors)
    }

    if total_rows > 0:
        metrics['rms'] = rmserror(preds, labels, total_rows)
    L.info(f"{metrics}")
    return np.array(errors), metrics

def evaluate_errors(errors):
    metrics = {
        'max': np.max(errors),
        '99th': np.percentile(errors, 99),
        '95th': np.percentile(errors, 95),
        '90th': np.percentile(errors, 90),
        'median': np.median(errors),
        'mean': np.mean(errors),
        'gmean': gmean(errors)
    }
    L.info(f"{metrics}")
    return metrics

def report_errors(dataset, result_file):
    df = pd.read_csv(RESULT_ROOT / dataset / result_file)
    evaluate_errors(df['error'])

def report_dynamic_errors(dataset, old_new_file, new_new_file, max_t, current_t):
    '''
    max_t: Time limit for update
    current_t: Model's update time.
    old_new_path: Result file of applying stale model on new workload
    new_new_path: Result file of applying updated model on new workload
    '''
    old_new_path = RESULT_ROOT / dataset / old_new_file
    new_new_path = RESULT_ROOT / dataset / new_new_file
    if max_t > current_t:
        try:
            o_n = pd.read_csv(old_new_path)
            n_n = pd.read_csv(new_new_path)
            assert len(o_n) == len(n_n), "In current version, the workload test size should be same."
            o_n_s = o_n.sample(frac = current_t / max_t)
            n_n_s = n_n.sample(frac = 1 - current_t / max_t)
            mixed_df = pd.concat([o_n_s, n_n_s], ignore_index=True, sort=False)
            return evaluate_errors(mixed_df['error'])
        except OSError:
            print('Cannot open file.')
    return -1

def lazy_derive(origin_result_file, result_file, r, labels):
    L.info("Already have the original result, directly derive the new prediction!")
    df = pd.read_csv(origin_result_file)
    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'error', 'predict', 'label', 'dur_ms'])
        for index, row in df.iterrows():
            p = np.round(row['predict'] * r)
            l = labels[index].cardinality
            writer.writerow([int(row['id']), qerror(p, l), p, l, row['dur_ms']])
    L.info("Done infering all predictions from previous result")

def run_test(dataset: str, version: str, workload: str, estimator: Estimator, overwrite: bool, lazy: bool=True, lw_vec=None, query_async=False) -> None:
    # for inference speed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    L.info(f"Start loading queryset:{workload} and labels for version {version} of dataset {dataset}...")
    # only keep test queries
    queries = load_queryset(dataset, workload)['test']
    labels = load_labels(dataset, version, workload)['test']

    if lw_vec is not None:
        X, gt = lw_vec
        #  assert isinstance(estimator, LWNN) or isinstance(estimator, LWTree), estimator
        assert len(X) == len(queries), len(X)
        assert np.array_equal(np.array([l.cardinality for l in labels]), gt)
        L.info("Hack for LW's method, use processed vector instead of raw query")
        queries = X

    # prepare file path, do not proceed if result already exists
    result_path = RESULT_ROOT / f"{dataset}"
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = result_path / f"{version}-{workload}-{estimator}.csv"
    if not overwrite and result_file.is_file():
        L.info(f"Already have the result {result_file}, do not run again!")
        exit(0)

    r = 1.0
    if version != estimator.table.version:
        test_row = load_table(dataset, version).row_num
        r = test_row / estimator.table.row_num
        L.info(f"Testing on a different data version, need to adjust the prediction according to the row number ratio {r} = {test_row} / {estimator.table.row_num}!")

        origin_result_file = RESULT_ROOT / dataset / f"{estimator.table.version}-{workload}-{estimator}.csv"
        if lazy and origin_result_file.is_file():
            return lazy_derive(origin_result_file, result_file, r, labels)

    if query_async:
        L.info("Start test estimator asynchronously...")
        for i, query in enumerate(queries):
            estimator.query_async(query, i)

        L.info('Waiting for queries to finish...')
        stats = ray.get([w.get_stats.remote() for w in estimator.workers])

        errors = []
        latencys = []
        with open(result_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'error', 'predict', 'label', 'dur_ms'])
            for i, label in enumerate(labels):
                r = stats[i%estimator.num_workers][i//estimator.num_workers]
                assert i == r.i, r
                error = qerror(r.est_card, label.cardinality)
                errors.append(error)
                latencys.append(r.dur_ms)
                writer.writerow([i, error, r.est_card, label.cardinality, r.dur_ms])

        L.info(f"Test finished, {np.mean(latencys)} ms/query in average")
        evaluate_errors(errors)
        return

    L.info("Start test estimator on test queries...")
    errors = []
    latencys = []
    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'error', 'predict', 'label', 'dur_ms'])
        for i, data in enumerate(zip(queries, labels)):
            query, label = data
            est_card, dur_ms = estimator.query(query)
            est_card = np.round(r * est_card)
            error = qerror(est_card, label.cardinality)
            errors.append(error)
            latencys.append(dur_ms)
            writer.writerow([i, error, est_card, label.cardinality, dur_ms])
            if (i+1) % 1000 == 0:
                L.info(f"{i+1} queries finished")
    L.info(f"Test finished, {np.mean(latencys)} ms/query in average")
    evaluate_errors(errors)


