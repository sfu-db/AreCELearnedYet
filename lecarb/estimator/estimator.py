import time
import logging
import numpy as np
from typing import Tuple, Any
from ..workload.workload import Query, query_2_triple
from ..dataset.dataset import Table

L = logging.getLogger(__name__)

class Estimator(object):
    """Base class for a cardinality estimator."""
    def __init__(self, table: Table, **kwargs: Any) -> None:
        self.table = table
        self.params = dict(kwargs)

    def __repr__(self) -> str:
        pstr = ';'.join([f"{p}={v}" for p, v in self.params.items()])
        return f"{self.__class__.__name__.lower()}-{pstr}"

    def query(self, query: Query) -> Tuple[float, float]:
        """return est_card, dur_ms"""
        raise NotImplementedError

def in_between(data: Any, val: Tuple[Any, Any]) -> bool:
    assert len(val) == 2
    lrange, rrange = val
    return np.greater_equal(data, lrange) & np.less_equal(data, rrange)

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '[]': in_between
}

class Oracle(Estimator):
    def __init__(self, table):
        super(Oracle, self).__init__(table=table)

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.table.row_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.table.data[c], v)
        card = bitmap.sum()
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

#  from pandasql import sqldf <- too slow
    #  def query(self, query):
    #      sql = query_2_sql(query, self.table)
    #      data = self.table.data
    #      start_stmp = time.time()
    #      df = sqldf(sql, locals())
    #      card = df.iloc[0, 0]
    #      dur_ms = (time.time() - start_stmp) * 1e3
    #      return card, dur_ms
