import time
import psycopg2
import logging
from typing import Any, Dict

from .estimator import Estimator
from .utils import run_test
from ..workload.workload import query_2_sql
from ..dataset.dataset import load_table
from ..constants import DATABASE_URL

L = logging.getLogger(__name__)

class Postgres(Estimator):
    def __init__(self, table, stat_target, seed):
        super(Postgres, self).__init__(table=table, version=table.version, stat=stat_target, seed=seed)

        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

        # construct statistics
        start_stmp = time.time()
        self.cursor.execute('select setseed({});'.format(1 / seed))
        for c in table.columns.values():
            self.cursor.execute('alter table \"{}\" alter column {} set statistics {};'.format(
                table.name, c.name, stat_target))
        self.cursor.execute('analyze \"{}\";'.format(self.table.name))
        self.conn.commit()
        dur_min = (time.time() - start_stmp) / 60

        # get size
        self.cursor.execute('select sum(pg_column_size(pg_stats)) from pg_stats where tablename=\'{}\''.format(self.table.name))
        size = self.cursor.fetchall()[0][0]
        #  self.cursor.execute('select sum(pg_column_size(pg_stats_ext)) from pg_stats_ext where tablename=\'{}\''.format(self.table.name))
        #  res = self.cursor.fetchall()[0][0]
        # might not have content in ext table
        #  if res is not None:
        #      size += res
        size = size / 1024 / 1024 # MB

        L.info(f"construct statistics finished, using {dur_min:.4f} minutes, All statistics consumes {size:.2f} MBs")

    def query(self, query):
        sql = 'explain(format json) {}'.format(query_2_sql(query, self.table, aggregate=False))
        #  L.info('sql: {}'.format(sql))

        start_stmp = time.time()
        self.cursor.execute(sql)
        dur_ms = (time.time() - start_stmp) * 1e3
        res = self.cursor.fetchall()
        card = res[0][0][0]['Plan']['Plan Rows']
        #  L.info(card)
        return card, dur_ms

    def query_sql(self, sql):
        sql = 'explain(format json) {}'.format(sql)
        #  L.info('sql: {}'.format(sql))

        start_stmp = time.time()
        self.cursor.execute(sql)
        res = self.cursor.fetchall()
        card = res[0][0][0]['Plan']['Plan Rows']
        #  L.info(card)
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

def test_postgres(seed: int, dataset: str, version: str, workload:str, params: Dict[str, Any], overwrite: bool):
    """
    params:
        version: the version of table that postgres construct statistics, might not be the same with the one we test on
        stat_target: size of the statistics limit
    """
    # prioriy: params['version'] (build statistics from another dataset) > version (build statistics on the same dataset)
    table = load_table(dataset, params.get('version') or version)

    L.info("construct postgres estimator...")
    estimator = Postgres(table, stat_target=params['stat_target'], seed=seed)
    L.info(f"built postgres estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)


