import time
import logging
from typing import Any, Dict
import psycopg2

from .estimator import Estimator
from .utils import run_test
from ..workload.workload import query_2_kde_sql, load_queryset
from ..dataset.dataset import load_table
from ..constants import KDE_DATABASE_URL

L = logging.getLogger(__name__)

class FeedbackKDE(Estimator):
    def __init__(self, table, ratio, train_num, seed):
        super(FeedbackKDE, self).__init__(table=table, version=table.version, ratio=ratio, train_num=train_num, seed=seed)
        self.sample_num = int(table.row_num * ratio)
        L.info(f"Going to collect {self.sample_num} samples")

        self.conn = psycopg2.connect(KDE_DATABASE_URL)
        self.conn.set_session('read uncommitted', autocommit=True)
        self.cursor = self.conn.cursor()

        # Make sure that debug mode is deactivated and that all model traces are removed (unless we want to reuse the model):
        self.cursor.execute(f"SELECT setseed({1/seed});")
        # self.cursor.execute("SET kde_debug TO true;")
        self.cursor.execute("SET kde_debug TO false;")
        self.cursor.execute("SET ocl_use_gpu TO true;")
        self.cursor.execute("SET kde_error_metric TO Quadratic;")

        # Remove all existing model traces if we don't reuse the model.
        self.cursor.execute("DELETE FROM pg_kdemodels;")
        self.cursor.execute("DELETE FROM pg_kdefeedback;")
        self.cursor.execute("SELECT pg_stat_reset();")

        # KDE-specific parameters.
        self.cursor.execute(f"SET kde_samplesize TO {self.sample_num};")
        self.cursor.execute("SET kde_enable TO true;")
        self.cursor.execute("SET kde_collect_feedback TO true;")

    def train_batch(self, queries):
        for i, query in enumerate(queries):
            self.cursor.execute(query_2_kde_sql(query, self.table))
            if (i + 1) % 100 == 0:
                L.info(f"{i+1} queries done")
        L.info("Finishing running all training queries")

        self.cursor.execute("SET kde_collect_feedback TO false;") # We don't need further feedback collection.
        self.cursor.execute("SET kde_enable_bandwidth_optimization TO true;")
        self.cursor.execute(f"SET kde_optimization_feedback_window TO {len(queries)};")

        stat_cnt = 100
        for c in self.table.columns.values():
            self.cursor.execute(f"alter table \"{self.table.name}\" alter column {c.name} set statistics {stat_cnt};")

        self.cursor.execute(f"analyze \"{self.table.name}\"({','.join(self.table.columns.keys())});")

        sample_file = f"/tmp/sample_{self.table.name}.csv"
        self.cursor.execute(f"SELECT kde_dump_sample('{self.table.name}', '{sample_file}');")

    def query(self, query):
        sql = f"explain(format json) {query_2_kde_sql(query, self.table)}"

        start_stmp = time.time()
        self.cursor.execute(sql)
        dur_ms = (time.time() - start_stmp) * 1e3
        res = self.cursor.fetchall()
        card = res[0][0][0]['Plan']['Plan Rows']
        #  L.info(card)
        return card, dur_ms

def test_kde(seed: int, dataset: str, version: str, workload:str, params: Dict[str, Any], overwrite: bool):
    """
    params:
        version: the version of table that postgres construct statistics, might not be the same with the one we test on
        ratio: ratio of the sample size
        train_num: number of queries use to train
    """
    # prioriy: params['version'] (build statistics from another dataset) > version (build statistics on the same dataset)
    table = load_table(dataset, params.get('version') or version)
    train_num = params['train_num']

    L.info("load training workload...")
    queries = load_queryset(dataset, workload)['train'][:train_num]

    L.info("construct postgres estimator...")
    estimator = FeedbackKDE(table, ratio=params['ratio'], train_num=train_num, seed=seed)

    L.info(f"start training with {train_num} queries...")
    start_stmp = time.time()
    estimator.train_batch(queries)
    dur_min = (time.time() - start_stmp) / 60
    L.info(f"built kde estimator: {estimator}, using {dur_min:1f} minutes")

    run_test(dataset, version, workload, estimator, overwrite)


