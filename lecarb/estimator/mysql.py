import time
import mysql.connector
import logging
from typing import Any, Dict
import numpy as np

from .estimator import Estimator
from .utils import run_test
from ..workload.workload import query_2_sql
from ..dataset.dataset import load_table
from ..constants import MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PSWD

L = logging.getLogger(__name__)

class MySQL(Estimator):
    def __init__(self, table, bucket, seed):
        super(MySQL, self).__init__(table=table, version=table.version, bucket=bucket, seed=seed)

        self.conn = mysql.connector.connect(user=MYSQL_USER, password=MYSQL_PSWD, host=MYSQL_HOST, port=MYSQL_PORT, database=MYSQL_DB)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

        # construct statistics
        start_stmp = time.time()
        self.cursor.execute(f"analyze table `{self.table.name}` update histogram on "
                            f"{','.join([c.name for c in table.columns.values()])} "
                            f"with {bucket} buckets;")
        rows = self.cursor.fetchall()
        L.info(f"{rows}")
        dur_min = (time.time() - start_stmp) / 60

        L.info(f"construct statistics finished, using {dur_min:.4f} minutes")

    def query(self, query):
        sql = 'explain {}'.format(query_2_sql(query, self.table, aggregate=False, dbms='mysql'))
        #  L.info('sql: {}'.format(sql))

        start_stmp = time.time()
        self.cursor.execute(sql)
        dur_ms = (time.time() - start_stmp) * 1e3
        res = self.cursor.fetchall()
        assert len(res) == 1, res
        # test 1
        card = np.round(0.01 * res[0][10] * self.table.row_num)
        # test 2
        #  card = np.round(0.01 * res[0][10] * res[0][9])
        #  L.info(card)
        return card, dur_ms

def test_mysql(seed: int, dataset: str, version: str, workload:str, params: Dict[str, Any], overwrite: bool):
    """
    params:
        version: the version of table that mysql construct statistics, might not be the same with the one we test on
        bucket: number of bucket for each histogram
    """
    # prioriy: params['version'] (build statistics from another dataset) > version (build statistics on the same dataset)
    table = load_table(dataset, params.get('version') or version)

    L.info("construct mysql estimator...")
    estimator = MySQL(table, params['bucket'], seed=seed)
    L.info(f"built mysql estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)


