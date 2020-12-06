import csv
from collections import OrderedDict
from typing import Dict, NamedTuple, Optional, Tuple, List, Any
import pickle
import numpy as np

from ..dtypes import is_categorical
from ..constants import DATA_ROOT, PKL_PROTO
from ..dataset.dataset import Table, load_table

class Query(NamedTuple):
    """predicate of each attritbute are conjunctive"""
    predicates: Dict[str, Optional[Tuple[str, Any]]]
    ncols: int

class Label(NamedTuple):
    cardinality: int
    selectivity: float

def new_query(table: Table, ncols) -> Query:
    return Query(predicates=OrderedDict.fromkeys(table.data.columns, None),
                 ncols=ncols)

def query_2_triple(query: Query, with_none: bool=True, split_range: bool=False
               ) -> Tuple[List[int], List[str], List[Any]]:
    """return 3 lists with same length: cols(columns names), ops(predicate operators), vals(predicate literals)"""
    cols = []
    ops = []
    vals = []
    for c, p in query.predicates.items():
        if p is not None:
            if split_range is True and p[0] == '[]':
                cols.append(c)
                ops.append('>=')
                vals.append(p[1][0])
                cols.append(c)
                ops.append('<=')
                vals.append(p[1][1])
            else:
                cols.append(c)
                ops.append(p[0])
                vals.append(p[1])
        elif with_none:
            cols.append(c)
            ops.append(None)
            vals.append(None)
    return cols, ops, vals

def query_2_sql(query: Query, table: Table, aggregate=True, split=False, dbms='postgres'):
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            val = f"\'{val}\'" if not isinstance(val, tuple) else tuple(f"\'{v}\'" for v in val)
        if op == '[]':
            if split:
                preds.append(f"{col} >= {val[0]}")
                preds.append(f"{col} <= {val[1]}")
            else:
                preds.append(f"({col} between {val[0]} and {val[1]})")
        else:
            preds.append(f"{col} {op} {val}")

    if dbms == 'mysql':
        return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM `{table.name}` WHERE {' AND '.join(preds)}"
    return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM \"{table.name}\" WHERE {' AND '.join(preds)}"

def query_2_kde_sql(query: Query, table: Table):
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            assert op =='=' and not isinstance(val, tuple), val
            val = table.columns[col].discretize(val).item()
        if op == '[]':
            preds.append(f"{col} >= {val[0]}")
            preds.append(f"{col} <= {val[1]}")
        else:
            preds.append(f"{col} {op} {val}")

    return f"SELECT * FROM \"{table.name}\" WHERE {' AND '.join(preds)}"

def query_2_deepdb_sql(query: Query, table: Table, aggregate=True, split=False):
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if op == '[]':
            val = table.columns[col].normalize(list(val))
            assert len(val) == 2, val
            if split:
                preds.append(f"{col} >= {val[0]}")
                preds.append(f"{col} <= {val[1]}")
            else:
                preds.append(f"({col} between {val[0]} and {val[1]})")
        else:
            val = table.columns[col].normalize(val).item()
            preds.append(f"{col} {op} {val}")

    return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM \"{table.name}\" WHERE {' AND '.join(preds)}"

def query_2_sqls(query: Query, table: Table):
    sqls = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            val = f"\'{val}\'" if not isinstance(val, tuple) else tuple(f"\'{v}\'" for v in val)

        if op == '[]':
            sqls.append(f"SELECT * FROM \"{table.name}\" WHERE {col} between {val[0]} and {val[1]}")
        else:
            sqls.append(f"SELECT * FROM \"{table.name}\" WHERE {col} {op} {val}")
    return sqls


def query_2_vector(query: Query, table: Table, upper: int=1):
    vec = []
    for col, pred in query.predicates.items():
        if pred is None:
            vec.extend([0.0, 1.0])
            continue
        op, val = pred
        if op == '[]':
            vec.extend([table.columns[col].normalize(val[0]).item(), table.columns[col].normalize(val[1]).item()])
        elif op == '>=':
            vec.extend([table.columns[col].normalize(val).item(), 1.0])
        elif op == '<=':
            vec.extend([0.0, table.columns[col].normalize(val).item()])
        elif op == '=':
            vec.extend([table.columns[col].normalize(val).item()] * 2)
        else:
            raise NotImplementedError
    return np.array(vec) * upper

def query_2_quicksel_vector(query: Query, table: Table, discrete_cols=set()):
    vec = []
    for col_name, pred in query.predicates.items():
        if pred is None:
            vec.extend([0.0, 1.0])
            continue
        op, val = pred
        col = table.columns[col_name]

        # adjust predicate to a proper range for discrete columns
        if col_name in discrete_cols:
            if is_categorical(col.dtype):
                val = col.discretize(val)
                minval = 0
                maxval = col.vocab_size
                vocab = np.arange(col.vocab_size)
            else: # integer values
                minval = col.minval
                maxval = col.maxval + 1
                vocab = col.vocab

            if op == '=':
                val = (val, val)
            elif op == '>=':
                val = (val, maxval)
            elif op == '<=':
                val = (minval, val)
            else:
                assert op == '[]'

            vocab = np.append(vocab, maxval)
            # argmax return 0 if no value in array satisfies
            val0 = vocab[np.argmax(vocab >= val[0])] if val[0] < maxval else maxval
            val1 = vocab[np.argmax(vocab > val[1])] if val[1] < maxval else maxval
            assert val0 <= val1, (val0, val1)
            assert val0 >= minval and val0 <= maxval, (val0, minval, maxval)
            assert val1 >= minval and val1 <= maxval, (val1, minval, maxval)
            # normalize to [0, 1]
            vec.extend([(val0-minval)/(maxval-minval), (val1-minval)/(maxval-minval)])

        # directly normalize continous columns
        else:
            if op == '>=':
                vec.extend([col.normalize(val).item(), 1.0])
            elif op == '<=':
                vec.extend([0.0, col.normalize(val).item()])
            elif op == '[]':
                vec.extend([col.normalize(val[0]).item(), col.normalize(val[1]).item()])
            else:
                raise NotImplementedError
    return np.array(vec)


def dump_queryset(dataset: str, name: str, queryset: Dict[str, List[Query]]) -> None:
    query_path = DATA_ROOT / dataset / "workload"
    query_path.mkdir(exist_ok=True)
    with open(query_path / f"{name}.pkl", 'wb') as f:
        pickle.dump(queryset, f, protocol=PKL_PROTO)

def load_queryset(dataset: str, name: str) -> Dict[str, List[Query]]:
    query_path = DATA_ROOT / dataset / "workload"
    with open(query_path / f"{name}.pkl", 'rb') as f:
        return pickle.load(f)

def dump_labels(dataset: str, version: str, name: str, labels: Dict[str, List[Label]]) -> None:
    label_path = DATA_ROOT / dataset / "workload"
    with open(label_path / f"{name}-{version}-label.pkl", 'wb') as f:
        pickle.dump(labels, f, protocol=PKL_PROTO)

def load_labels(dataset: str, version: str, name: str) -> Dict[str, List[Label]]:
    label_path = DATA_ROOT / dataset / "workload"
    with open(label_path / f"{name}-{version}-label.pkl", 'rb') as f:
        return pickle.load(f)

def dump_sqls(dataset: str, version: str, workload: str, group: str='test'):
    table = load_table(dataset, version)
    queryset = load_queryset(dataset, workload)
    labels = load_labels(dataset, version, workload)

    with open('test.csv', 'w') as f:
        writer = csv.writer(f)
        for query, label in zip(queryset[group], labels[group]):
            sql = query_2_sql(query, table, aggregate=False, dbms='sqlserver')
            writer.writerow([sql, label.cardinality])
