import enum
import copy
import logging
import pickle
import time
import threading
import bisect
from typing import Any, Dict, Tuple
import numpy as np

from .estimator import Estimator
from .utils import run_test
from ..constants import MODEL_ROOT, NUM_THREADS, PKL_PROTO
from ..dtypes import is_categorical
from ..dataset.dataset import load_table
from ..workload.workload import query_2_triple

L = logging.getLogger(__name__)

class M(enum.IntEnum):
    LEFT = 0
    RIGHT = 1
    LEFT_IN = 2
    SPREAD = 3
    DISTINCT = 4

class Partition(object):
    def __init__(self, pid=0):
        self.pid = pid

        # for inference
        self.meta = [] # (left, right, include_left, spread_length, distinct)
        self.density = None

        # for construct
        self.data = None
        self.maxdiff = None # (area size, column, value)

    def __str__(self):
        if self.density is None:
            return f'{self.pid}: # : {len(self.data)}\nMaxDiff: {self.maxdiff}\nMetadata: {self.meta}'
        else:
            return f'{self.pid}: Density: {self.density}\nMetadata: {self.meta}'

    def clean(self):
        self.data = None
        self.maxdiff = None

    def construct_from_table(self, table):
        # normalize is better than only digitalize data
        self.data = table.normalize()
        #  self.data = table.digitalize()
        for c in self.data.columns:
            self.meta.append([self.data[c].min(), self.data[c].max(), True, None, None])

    def get_maxdiff(self):
        # only need to calculate split point once if data dose not change
        if self.maxdiff is not None:
            return self.maxdiff[0]

        for cid, c in enumerate(self.data.columns):
            counter = self.data[c].value_counts().sort_index()
            areas = counter.iloc[:-1] * (counter.index[1:] - counter.index[:-1])
            if len(areas) > 0:
                c_max = areas.max()
                if self.maxdiff is None or c_max > self.maxdiff[0]:
                    self.maxdiff = (c_max, cid, areas.idxmax())

        if self.maxdiff is None:
            self.maxdiff = (0, None, None)
        return self.maxdiff[0]

    def split_partition(self):
        if self.maxdiff is None:
            self.get_maxdiff()
        assert self.maxdiff is not None and self.maxdiff[0] > 0

        _, cid, split = self.maxdiff
        c = self.data.columns[cid]

        p1 = Partition()
        p1.data = self.data[self.data[c] <= split]
        p1.meta = copy.deepcopy(self.meta)
        p1.meta[cid] = [p1.meta[cid][0], split, p1.meta[cid][2], None, None]

        p2 = Partition()
        p2.data = self.data[self.data[c] > split]
        p2.meta = copy.deepcopy(self.meta)
        p2.meta[cid] = [split, p2.meta[cid][1], False, None, None]

        return p1, p2

    def calculate_spread_density(self):
        total_distinct = 1 # product of # distinct of each column
        #  L.error(f'total data: {len(self.data)}')
        for cid in range(len(self.meta)):
            c = self.data.columns[cid]
            unique = self.data[c].unique()
            distinct = len(unique)
            self.meta[cid][M.DISTINCT] = distinct
            #  L.error(f'column: {cid}, distinct: {distinct}')
            if distinct == 1:
                self.meta[cid][M.SPREAD] = float(unique.item() - self.meta[cid][M.LEFT])
                continue
            if self.meta[cid][M.LEFT_IN] is True:
                self.meta[cid][M.SPREAD] = float((self.meta[cid][M.RIGHT] - self.meta[cid][M.LEFT]) / (distinct - 1))
            else:
                self.meta[cid][M.SPREAD] = float((self.meta[cid][M.RIGHT] - self.meta[cid][M.LEFT]) / distinct)
            total_distinct *= distinct
        self.density = len(self.data) / total_distinct
        #  L.error(self)

    def query(self, columns, operators, values):
        def get_points_on_left(v, closed=False):
            if v < left or (v == left and (not closed)):
                return 0
            if v > right or (v == right and closed):
                return distinct
            covered, remains = divmod((v - left), spread)
            if not closed and remains < 1e-10:
                covered -= 1
            covered = int(covered) + 1
            return covered

        #  L.error('')
        #  L.error(self)
        #  L.error('')

        total_covered = 1
        for cid, op, val in zip(columns, operators, values):
            left, right, left_in, spread, distinct = self.meta[cid]

            # if only has one point, get the true value
            if distinct == 1:
                left = right = left + spread
            elif not left_in:
                left += spread

            assert left <= right, f'{self.pid}-{cid}: {self.meta[cid]}'

            c_covered = None
            if op == '<':
                c_covered = get_points_on_left(val, closed=False)
            elif op == '<=':
                c_covered = get_points_on_left(val, closed=True)
            elif op == '>':
                c_covered = distinct - get_points_on_left(val, closed=True)
            elif op == '>=':
                c_covered = distinct - get_points_on_left(val, closed=False)
            elif op == '[]':
                # (>= val[0] and <= val[1]) -> (<= val[1]) - (< val[0])
                c_covered = get_points_on_left(val[1], closed=True)
                if c_covered > 0:
                    c_covered -= get_points_on_left(val[0], closed=False)
            elif op == '=':
                if val < left or val > right:
                    c_covered = 0
                else:
                    # if equal, cover 1 value
                    c_covered = 1
                # just like mentioned in Naru, it tends to underestimate
                #  elif (spread == 0 and val == right) or (val - left) % spread == 0:
                #      c_covered = 1
                #  else:
                #      c_covered = 0

            assert type(c_covered) == int and c_covered >= 0, f'{self.pid}-{cid}-{op}-{val}:{self.meta[cid]}, c_cover: {c_covered}'
            total_covered *= c_covered
            if total_covered == 0:
                break

        if total_covered == 0:
            return 0

        for cid in range(len(self.meta)):
            if not cid in columns:
                total_covered *= self.meta[cid][M.DISTINCT]
        return total_covered * self.density

class Estimation(object):
    def __init__(self, num_part, num_threads):
        self.card = np.zeros((num_threads))
        self.parts = np.array(range(num_threads)) * int(num_part / num_threads)
        self.parts = np.append(self.parts, num_part)

class MHist(Estimator):
    def __init__(self, partitions, table):
        super(MHist, self).__init__(table=table, bins=len(partitions))
        self.partitions = partitions

        # index for faster inference (refer from Naru)
        # map<cid, map<bound_type, map<bound_value, list(partition id)>>>
        self.column_bound_map = {}
        for cid in range(self.table.col_num):
            self.column_bound_map[cid] = {}
            self.column_bound_map[cid]['l'] = {}
            self.column_bound_map[cid]['u'] = {}
        # map<cid, map<bound_type, sorted_list(bound_value)>>
        self.column_bound_index = {}
        for cid in range(self.table.col_num):
            self.column_bound_index[cid] = {}
            self.column_bound_index[cid]['l'] = []
            self.column_bound_index[cid]['u'] = []
        self._build_index()

    def _build_index(self):
        for cid in range(self.table.col_num):
            for pid, p in enumerate(self.partitions):
                if p.meta[cid][M.LEFT] not in self.column_bound_map[cid]['l']:
                    self.column_bound_map[cid]['l'][p.meta[cid][M.LEFT]] = [pid]
                else:
                    self.column_bound_map[cid]['l'][p.meta[cid][M.LEFT]].append(pid)

                if p.meta[cid][M.RIGHT] not in self.column_bound_map[cid]['u']:
                    self.column_bound_map[cid]['u'][p.meta[cid][M.RIGHT]] = [pid]
                else:
                    self.column_bound_map[cid]['u'][p.meta[cid][M.RIGHT]].append(pid)

                self.column_bound_index[cid]['l'].append(p.meta[cid][M.LEFT])
                self.column_bound_index[cid]['u'].append(p.meta[cid][M.RIGHT])
            self.column_bound_index[cid]['l'] = sorted(set(self.column_bound_index[cid]['l']))
            self.column_bound_index[cid]['u'] = sorted(set(self.column_bound_index[cid]['u']))

    def _get_valid_pids(self, cid, op, val):
        if op in ['<', '<=']:
            valid_set = set()
            if op == '<':
                insert_index = bisect.bisect_left(self.column_bound_index[cid]['l'], val)
                for i in range(insert_index):
                    valid_set = valid_set.union(self.column_bound_map[cid]['l'][self.column_bound_index[cid]['l'][i]])
            else:
                insert_index = bisect.bisect(self.column_bound_index[cid]['l'], val)
                for i in range(insert_index):
                    if self.column_bound_index[cid]['l'][i] == val:
                        for pid in self.column_bound_map[cid]['l'][val]:
                            if self.partitions[pid].meta[cid][M.LEFT_IN]:
                                # add only when the lower bound is inclusive
                                valid_set.add(pid)
                    else:
                        valid_set = valid_set.union(self.column_bound_map[cid]['l'][self.column_bound_index[cid]['l'][i]])
            return valid_set

        if op in ['>', '>=']:
            valid_set = set()
            insert_index = None
            if op == '>':
                insert_index = bisect.bisect(self.column_bound_index[cid]['u'], val)
            else:
                insert_index = bisect.bisect_left(self.column_bound_index[cid]['u'], val)
            for i in range(insert_index, len(self.column_bound_index[cid]['u'])):
                valid_set = valid_set.union(self.column_bound_map[cid]['u'][self.column_bound_index[cid]['u'][i]])
            return valid_set

        assert op in ['=', '[]'], op
        lower_v, upper_v = val if type(val) is tuple else (val, val)
        lower_bound_set = set()
        insert_index = bisect.bisect(self.column_bound_index[cid]['l'], upper_v)
        for i in range(insert_index):
            if self.column_bound_index[cid]['l'][i] == upper_v:
                for pid in self.column_bound_map[cid]['l'][upper_v]:
                    if self.partitions[pid].meta[cid][M.LEFT_IN]:
                        # add only when the lower bound is inclusive
                        lower_bound_set.add(pid)
            else:
                lower_bound_set = lower_bound_set.union(
                    self.column_bound_map[cid]['l'][
                        self.column_bound_index[cid]['l'][i]])

        upper_bound_set = set()
        insert_index = bisect.bisect_left(self.column_bound_index[cid]['u'], lower_v)
        for i in range(insert_index, len(self.column_bound_index[cid]['u'])):
            upper_bound_set = upper_bound_set.union(
                self.column_bound_map[cid]['u'][self.column_bound_index[cid]['u'][i]])
        return lower_bound_set.intersection(upper_bound_set)

    def query_worker(self, tid, est, columns, operators, values, candidate_pids):
        for i in range(est.parts[tid], est.parts[tid+1]):
            est.card[tid] += self.partitions[candidate_pids[i]].query(columns, operators, values)

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        # descritize predicate parameters for non-numerical columns
        for i, predicate in enumerate(zip(columns, operators, values)):
            cname, op, val = predicate
            col = self.table.columns[cname]
            val = col.normalize(list(val) if type(val) is tuple else [val])
            values[i] = tuple(val) if len(val) > 1 else val.item()
            #  if is_categorical(col.dtype):
            #      val = col.discretize(list(val) if type(val) is tuple else [val])
            #      values[i] = tuple(val) if len(val) > 1 else val.item()
        # convert column names to indices
        columns = [self.table.data.columns.get_loc(c) for c in columns]
        start_stmp = time.time()

        # use index to find related partition ids
        candidate_pids = set(range(len(self.partitions)))
        for cid, op, val in zip(columns, operators, values):
            candidate_pids = candidate_pids.intersection(self._get_valid_pids(cid, op, val))

        #  query on each partition
        #  candidate_pids = list(candidate_pids)
        #  num_threads = NUM_THREADS if len(candidate_pids) > (NUM_THREADS * 10) else 1
        #  est = Estimation(len(candidate_pids), num_threads)
        #  for i in range(num_threads):
        #      t = threading.Thread(target=self.query_worker, args=(i, est, columns, operators, values, candidate_pids))
        #      t.start()

        #  main_thread = threading.currentThread()
        #  for t in threading.enumerate():
        #      if t is not main_thread:
        #          t.join()

        est_card = []
        for pid in candidate_pids:
            est_card.append(self.partitions[pid].query(columns, operators, values))

        dur_ms = (time.time() - start_stmp) * 1e3

        #  return np.round(est.card.sum()), dur_ms
        return np.round(np.sum(est_card)), dur_ms

def get_partition_num(col_num, size_limit_mb):
    # for each partition, we need record the follow information for each column:
    # density, left, right, spread_length: 4 bytes for each
    # include_left: 1 byte
    # do not count total of row number here since all method need to record this
    # 13 = 3 * 4 + 1
    # 17 = 4 * 4 + 1
    return int((size_limit_mb * 1024 * 1024) // (4 + col_num * 13))
    #  return int((size_limit_mb * 1024 * 1024) // (4 + col_num * 17))

def get_hist_size(num_bins, col_num):
    # for each partition, we need record the follow information for each column:
    # density, left, right, spread_length: 4 bytes for each
    # include_left: 1 byte
    # do not count total of row number here since all method need to record this
    return (num_bins * (col_num * 13 + 4)) / 1024 / 1024
    #  return (num_bins * (col_num * 17 + 4)) / 1024 / 1024

def print_partitions(partitions):
    L.info('')
    for p in partitions:
        L.info(f'\n{p}')
    L.info('======================')

def construct_maxdiff(table, num_bins):
    partitions = []

    start_stmp = time.time()
    for i in range(num_bins):
        if len(partitions) == 0:
            partitions.append(Partition())
            partitions[0].construct_from_table(table)
            continue

        # find the partition has maxdiff to split
        maxdiff = 0
        pid = None
        for i, p in enumerate(partitions):
            p_md = p.get_maxdiff()
            if p_md > maxdiff:
                maxdiff = p_md
                pid = i

        #  print_partitions(partitions)
        if maxdiff == 0:
            L.info('Maxdiff is 0 before reach partition limit!')
            break

        p = partitions.pop(pid)
        p1, p2 = p.split_partition()
        partitions.extend([p1, p2])

        if (i+1) % 100 == 0:
            L.info(f'Constructed {i+1} partitions!')

    for p in partitions:
        p.calculate_spread_density()
        p.clean()
    hist_size = get_hist_size(len(partitions), len(table.columns))
    #  print_partitions(partitions)

    dur_min = (time.time() - start_stmp) / 60
    L.info(f'Construct MaxDiff Hist (MHIST-2) finished, use {len(partitions)} partitions ({hist_size:.2f}MB)! Time spent since start: {dur_min:.2f} mins')

    state = {
        'device': 'cpu',
        'threads': NUM_THREADS,
        'dataset': table.dataset,
        'version': table.version,
        'partitions': partitions,
        'train_time': dur_min,
        'model_size': hist_size,
    }
    return state

def load_mhist(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset / f"{model_name}.pkl"
    L.info(f"load model from {model_file} ...")
    with open(model_file, 'rb') as f:
        state = pickle.load(f)

    table = load_table(dataset, state['version'])
    partitions = state['partitions']
    #  print_partitions(partitions)
    estimator = MHist(partitions, table)
    return estimator, state

def test_mhist(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the histogram is built from, might not be the same with the one we test on
        num_bins: maximum number of partitions
    """
    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    table = load_table(dataset, params.get('version') or version)

    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-mhist_bin{params['num_bins']}.pkl"

    if model_file.is_file():
        L.info(f"{model_file} already exists, directly load and use")
        with open(model_file, 'rb') as f:
            state = pickle.load(f)
    else:
        L.info(f"Construct MHist with at most {params['num_bins']} bins...")
        state = construct_maxdiff(table, params['num_bins'])
        with open(model_file, 'wb') as f:
            pickle.dump(state, f, protocol=PKL_PROTO)
        L.info(f"MHist saved to {model_file}")

    partitions = state['partitions']
    #  print_partitions(partitions)
    estimator = MHist(partitions, table)
    L.info(f"Built MHist estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)
