import time
import copy
import json
import pickle
import logging
import collections
from typing import Any, Dict, NamedTuple
import numpy as np
import pandas as pd
from .estimator import Estimator, OPS
from .utils import run_test
from ..workload.workload import query_2_triple
from ..dataset.dataset import load_table
from ..constants import NUM_THREADS
from ..dtypes import is_categorical

L = logging.getLogger(__name__)

"""The below implementation is copied from https://github.com/naru-project/naru with some slight modification"""

class BayesianNetworkWorker(object):
    """Progressive sampling with a pomegranate bayes net."""

    def build_discrete_mapping(self, table, discretize, discretize_method):
        assert discretize_method in ["equal_size",
                                     "equal_freq"], discretize_method
        self.max_val = collections.defaultdict(lambda: None)
        if not discretize:
            return {}
        table = table.copy()
        mapping = {}
        for col_id in range(len(table[0])):
            col = table[:, col_id]
            if max(col) > discretize:
                if discretize_method == "equal_size":
                    denom = (max(col) + 1) / discretize
                    fn = lambda v: np.floor(v / denom)
                elif discretize_method == "equal_freq":
                    per_bin = len(col) // discretize
                    counts = collections.defaultdict(int)
                    for x in col:
                        counts[int(x)] += 1
                    assignments = {}
                    i = 0
                    bin_size = 0
                    for k, count in sorted(counts.items()):
                        if bin_size > 0 and bin_size + count >= per_bin:
                            bin_size = 0
                            i += 1
                        assignments[k] = i
                        self.max_val[col_id] = i
                        bin_size += count
                    assignments = np.array(
                        [assignments[i] for i in range(int(max(col) + 1))])

                    def capture(assignments):

                        def fn(v):
                            return assignments[v.astype(np.int32)]

                        return fn

                    fn = capture(assignments)
                else:
                    assert False

                mapping[col_id] = fn
        return mapping

    def apply_discrete_mapping(self, table, discrete_mapping):
        table = table.copy()
        for col_id in range(len(table[0])):
            if col_id in discrete_mapping:
                fn = discrete_mapping[col_id]
                table[:, col_id] = fn(table[:, col_id])
        return table

    def apply_discrete_mapping_to_value(self, value, col_id, discrete_mapping):
        if col_id not in discrete_mapping:
            return value
        return discrete_mapping[col_id](value)

    def __init__(self,
                 #  dataset,
                 table,
                 num_samples,
                 algorithm="greedy",
                 max_parents=-1,
                 topological_sampling_order=True,
                 use_pgm=True,
                 discretize=None,
                 discretize_method="equal_size",
                 root=None):

        from pomegranate import BayesianNetwork
        self.discretize = discretize
        self.discretize_method = discretize_method
        self.table = copy.deepcopy(table)
        self.dataset = np.stack([col.discretize(self.table.data[cname]) for cname, col in self.table.columns.items()], axis=1)
        self.algorithm = algorithm
        self.topological_sampling_order = topological_sampling_order
        self.num_samples = num_samples
        self.discrete_mapping = self.build_discrete_mapping(
            self.dataset, discretize, discretize_method)
        self.discrete_table = self.apply_discrete_mapping(
            self.dataset, self.discrete_mapping)
        L.info('calling BayesianNetwork.from_samples...')
        t = time.time()
        self.model = BayesianNetwork.from_samples(self.discrete_table,
                                                  algorithm=self.algorithm,
                                                  max_parents=max_parents,
                                                  n_jobs=NUM_THREADS,
                                                  root=root)
        L.info(f'done! took {(time.time() - t)/60:.2f} mins')

        def size(states):
            n = 0
            for state in states:
                if "distribution" in state:
                    dist = state["distribution"]
                else:
                    dist = state
                if dist["name"] == "DiscreteDistribution":
                    for p in dist["parameters"]:
                        n += len(p)
                elif dist["name"] == "ConditionalProbabilityTable":
                    for t in dist["table"]:
                        n += len(t)
                    if "parents" in dist:
                        for parent in dist["parents"]:
                            n += size(dist["parents"])
                else:
                    assert False, dist["name"]
            return n

        self.size = 4 * size(json.loads(self.model.to_json())["states"])
        L.info(f'model size is {self.size/1024/1024:.2f}MB')

        # print('json:\n', self.model.to_json())
        self.json_size = len(self.model.to_json())
        self.use_pgm = use_pgm
        #        print(self.model.to_json())

        if topological_sampling_order:
            self.sampling_order = []
            while len(self.sampling_order) < len(self.model.structure):
                for i, deps in enumerate(self.model.structure):
                    if i in self.sampling_order:
                        continue  # already ordered
                    if all(d in self.sampling_order for d in deps):
                        self.sampling_order.append(i)
                L.debug(f"Building sampling order {self.sampling_order}")
        else:
            self.sampling_order = list(range(len(self.model.structure)))
        L.info(f"Using sampling order {self.sampling_order} {str(self)}")

        if use_pgm:
            from pgmpy.models import BayesianModel
            data = pd.DataFrame(self.discrete_table.astype(np.int64))
            spec = []
            orphans = []
            for i, parents in enumerate(self.model.structure):
                for p in parents:
                    spec.append((p, i))
                if not parents:
                    orphans.append(i)
            L.info(f"Model spec {spec}")
            model = BayesianModel(spec)
            for o in orphans:
                model.add_node(o)
            L.info('calling pgm.BayesianModel.fit...')
            t = time.time()
            model.fit(data)
            L.info(f'done! took {(time.time() - t)/60:.2f} mins')
            self.pgm_model = model

    def __str__(self):
        return "bn-{}-{}-{}-{}-bytes-{}-{}-{}".format(
            self.algorithm,
            self.num_samples,
            "topo" if self.topological_sampling_order else "nat",
            self.size,
            # self.json_size,
            self.discretize,
            self.discretize_method if self.discretize else "na",
            "pgmpy" if self.use_pgm else "pomegranate")

    def Query(self, query):
        columns, operators, vals = query_2_triple(query, with_none=True)

        start_stmp = time.time()
        ncols = len(columns)
        nrows = self.discrete_table.shape[0]
        assert ncols == self.discrete_table.shape[1], (
            ncols, self.discrete_table.shape)

        def adjust_literals(col_id, op, val):
            col = list(self.table.columns.values())[col_id]
            if is_categorical(col.dtype):
                return col.discretize([val])[0]
            if op == '>=':
                assert val <= col.maxval, (col.name, val, col.maxval)
                val = col.vocab[np.argmax(col.vocab >= val)]
                return col.discretize([val])[0]
            elif op == '<=':
                assert val >= col.minval, (col.name, val, col.minval)
                val = col.vocab[::-1][np.argmax(col.vocab[::-1] <= val)]
                return col.discretize([val])[0]
            elif op == '[]':
                assert val[0] <= col.maxval, (col.name, val[0], col.maxval)
                assert val[1] >= col.minval, (col.name, val[1], col.minval)
                val0 = col.vocab[np.argmax(col.vocab >= val[0])]
                val1 = col.vocab[::-1][np.argmax(col.vocab[::-1] <= val[1])]
                return col.discretize([val0, val1])
            elif op == '=':
                assert val in col.vocab
                return col.discretize([val])[0]
            else:
                L.error(f"unknown operator: {op}")
                raise NotImplementedError

        def draw_conditional_pgm(evidence, col_id):
            """PGM version of draw_conditional()"""

            if operators[col_id] is None:
                op = None
                val = None
            else:
                op = OPS[operators[col_id]]
                val = adjust_literals(col_id, operators[col_id], vals[col_id])
                if operators[col_id] == '[]':
                    val = [self.apply_discrete_mapping_to_value(v, col_id, self.discrete_mapping) for v in val]
                else:
                    val = self.apply_discrete_mapping_to_value(val, col_id, self.discrete_mapping)
                if self.discretize:
                    # avoid some bad cases
                    if operators[col_id] == "<" and val == 0:
                        val += 1
                    elif operators[col_id] == ">" and val == self.max_val[col_id]:
                        val -= 1

            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                for k, v in enumerate(distribution):
                    if op(k, val):
                        p += v
                return p

            from pgmpy.inference import VariableElimination
            model_inference = VariableElimination(self.pgm_model)
            xi_distribution = []
            for row in evidence:
                e = {}
                for i, v in enumerate(row):
                    if v is not None:
                        e[i] = v
                result = model_inference.query(variables=[col_id], evidence=e)
                xi_distribution.append(result[col_id].values)

            xi_marginal = [prob_match(d) for d in xi_distribution]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in enumerate(d):
                    if not op or op(k, val):
                        keys.append(k)
                        prob.append(p)
                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        def draw_conditional(evidence, col_id):
            """Draws a new value x_i for the column, and returns P(x_i|prev).
            Arguments:
                evidence: shape [BATCH, ncols] with None for unknown cols
                col_id: index of the current column, i
            Returns:
                xi_marginal: P(x_i|x0...x_{i-1}), computed by marginalizing
                    across the range constraint
                match_rows: the subset of rows from filtered_rows that also
                    satisfy the predicate at column i.
            """

            if operators[col_id] is None:
                op = None
                val = None
            else:
                op = OPS[operators[col_id]]
                val = adjust_literals(col_id, operators[col_id], vals[col_id])
                if operators[col_id] == '[]':
                    val = [self.apply_discrete_mapping_to_value(v, col_id, self.discrete_mapping) for v in val]
                else:
                    val = self.apply_discrete_mapping_to_value(val, col_id, self.discrete_mapping)
                if self.discretize:
                    # avoid some bad cases
                    if val == 0 and operators[col_id] == "<":
                        val += 1
                    elif val == self.max_val[col_id] and operators[
                            col_id] == ">":
                        val -= 1

            def prob_match(distribution):
                if not op:
                    return 1.
                p = 0.
                for k, v in distribution.items():
                    if op(k, val):
                        p += v
                return p

            xi_distribution = self.model.predict_proba(evidence,
                                                       max_iterations=1,
                                                       n_jobs=-1)
            xi_marginal = [
                prob_match(d[col_id].parameters[0]) for d in xi_distribution
            ]
            filtered_distributions = []
            for d in xi_distribution:
                keys = []
                prob = []
                for k, p in d[col_id].parameters[0].items():
                    if not op or op(k, val):
                        keys.append(k)
                        prob.append(p)
                denominator = sum(prob)
                if denominator == 0:
                    prob = [1. for _ in prob]  # doesn't matter
                    if len(prob) == 0:
                        prob = [1.]
                        keys = [0.]
                prob = np.array(prob) / sum(prob)
                filtered_distributions.append((keys, prob))
            xi_samples = [
                np.random.choice(k, p=v) for k, v in filtered_distributions
            ]

            return xi_marginal, xi_samples

        p_estimates = [1. for _ in range(self.num_samples)]
        evidence = [[None] * ncols for _ in range(self.num_samples)]
        for col_id in self.sampling_order:
            if self.use_pgm:
                xi_marginal, xi_samples = draw_conditional_pgm(evidence, col_id)
            else:
                xi_marginal, xi_samples = draw_conditional(evidence, col_id)
            for ev_list, xi in zip(evidence, xi_samples):
                ev_list[col_id] = xi
            for i in range(self.num_samples):
                p_estimates[i] *= xi_marginal[i]

        dur_ms = (time.time() - start_stmp) * 1e3
        return np.round(np.mean(p_estimates) * nrows).astype(dtype=np.int32, copy=False), dur_ms

class Result(NamedTuple):
    i: int
    est_card: int
    dur_ms: float

class Bayes(Estimator):
    def __init__(self, table, samples, discretize, parallelism):
        super(Bayes, self).__init__(table=table, version=table.version, samples=samples, discretize=discretize)
        self.num_workers = parallelism
        self.workers = []
        self.start_workers(parallelism)

    def start_workers(self, parallelism):
        import ray
        ray.init(redis_password='xxx')

        @ray.remote
        class Worker(object):
            def __init__(self, table, samples, discretize, i):
                self.estimator = BayesianNetworkWorker(table,
                                                       samples,
                                                       'chow-liu',
                                                       topological_sampling_order=True,
                                                       root=0,
                                                       max_parents=2,
                                                       use_pgm=False,
                                                       discretize=discretize,
                                                       discretize_method='equal_freq')
                self.i = i
                self.stats = []

            def run_query(self, query, j):
                query = pickle.loads(query)
                card, dur_ms = self.estimator.Query(query)
                self.stats.append(Result(i=j, est_card=card, dur_ms=dur_ms))
                if (j+1) % 10 == 0:
                    L.info(f'Finished {j+1} queries')

            def get_stats(self):
                return self.stats

        L.info(f"construct {parallelism} bayesian network workers...")
        for i in range(parallelism):
            self.workers.append(Worker.remote(self.table, self.params['samples'], self.params['discretize'], i))

    def query(self, query):
        pass

    def query_async(self, query, i):
        self.workers[i % self.num_workers].run_query.remote(pickle.dumps(query), i)

def test_bayesnet(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the bayesian network is built from, might not be the same with the one we test on
        samples: # progressive samples of each inference
        discretize: # bins for each column
        parallelism: # threads to inference in parallel
    """
    np.random.seed(seed)

    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    table = load_table(dataset, params.get('version') or version)

    estimator = Bayes(table, samples=params['samples'], discretize=params['discretize'], parallelism=params['parallelism'])
    L.info(f"built bayesian network estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite, query_async=True)
