import csv
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .workload import load_queryset, load_labels, query_2_quicksel_vector, new_query
from ..dtypes import is_discrete, is_categorical
from ..dataset.dataset import load_table
from ..estimator.estimator import Oracle
from ..constants import DATA_ROOT

L = logging.getLogger(__name__)

def dump_quicksel_query_files(dataset: str, version: str, workload: str, overwrite: bool) -> None:
    result_path = DATA_ROOT / dataset / "quicksel"
    result_path.mkdir(exist_ok=True)
    if not overwrite and Path(result_path / f"{workload}-{version}-train.csv").is_file() and Path(result_path / f"{workload}-{version}-test.csv").is_file():
        L.info("Already has quicksel workload file dumped, do not continue")
        return

    table = load_table(dataset, version)
    queryset = load_queryset(dataset, workload)
    labels = load_labels(dataset, version, workload)

    discrete_cols = set()
    for col_name, col in table.columns.items():
        # hard code for power dataset since all these columns are actually integers
        if dataset[:5] == 'power' and col_name in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
            discrete_cols.add(col_name)
            continue
        if is_discrete(col.dtype):
            discrete_cols.add(col_name)
    L.info(f"Detect discrete columns: {discrete_cols}")

    for group in ('train', 'test'):
        L.info(f"Start dump {workload} for {dataset}-{version}")
        result_file = result_path / f"{workload}-{version}-{group}.csv"
        with open(result_file, 'w') as f:
            writer = csv.writer(f)
            for query, label in zip(queryset[group], labels[group]):
                vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                vec.append(label.selectivity)
                writer.writerow(vec)
        L.info(f"File dumped to {result_file}")

def generate_quicksel_permanent_assertions(dataset: str, version: str, params: Dict[str, Dict[str, Any]], overwrite: bool) -> None:
    result_path = DATA_ROOT / dataset / "quicksel"
    result_path.mkdir(exist_ok=True)
    result_file = result_path / f"{version}-permanent.csv"
    if not overwrite and result_file.is_file():
        L.info("Already has permanent assertions generated, do not continue")
        return

    count = params['count']+1

    table = load_table(dataset, version)
    oracle = Oracle(table)

    discrete_cols = set()
    for col_name, col in table.columns.items():
        # hard code for power dataset since all these columns are actually integers
        if dataset[:5] == 'power' and col_name in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
            discrete_cols.add(col_name)
            continue
        if is_discrete(col.dtype):
            discrete_cols.add(col_name)
    L.info(f"Detect discrete columns: {discrete_cols}")

    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([0.0, 1.0] * table.col_num + [1.0])
        for col_id, col in enumerate(table.columns.values()):
            L.info(f"Start generate permanent queries on column {col.name}")
            # hard code for power dataset since all these columns are actually integers
            if is_discrete(col.dtype) or (dataset[:5] == 'power' and col.name in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']):
                if is_categorical(col.dtype):
                    L.info("Categorical column")
                    if col.vocab_size <= count:
                        for i in range(col.vocab_size):
                            query = new_query(table, ncols=1)
                            query.predicates[col.name] = ('=', col.vocab[i])
                            card, _ = oracle.query(query)
                            #  vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                            #  vec.append(card/table.row_num)
                            vec = [0.0, 1.0] * table.col_num
                            vec.append(card/table.row_num)
                            vec[col_id*2] = i/col.vocab_size
                            vec[col_id*2+1] = (i+1)/col.vocab_size
                            writer.writerow(vec)
                            L.info(f"# {i}: {query.predicates[col.name]}, card={card}\n\t{vec}")
                    else:
                        minval = 0
                        maxval = col.vocab_size
                        norm_range = np.linspace(0.0, 1.0, count, dtype=np.float32)
                        prange = minval + (maxval - minval) * norm_range
                        for i in range(len(prange)-1):
                            val0 = col.vocab[np.ceil(prange[i]).astype(int)]
                            val1 = col.vocab[np.ceil(prange[i+1]).astype(int)-1]
                            assert np.greater_equal(np.array(val1).astype(object), val0), (val1, val0)
                            query = new_query(table, ncols=1)
                            query.predicates[col.name] = ('[]', (val0, val1))
                            card, _ = oracle.query(query)
                            #  vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                            #  vec.append(card/table.row_num)

                            vec = [0.0, 1.0] * table.col_num
                            vec.append(card/table.row_num)
                            vec[col_id*2] = norm_range[i]
                            vec[col_id*2+1] = norm_range[i+1]
                            writer.writerow(vec)
                            L.info(f"# {i}: {query.predicates[col.name]}, card={card}\n\t{vec}")
                else:
                    L.info("Integer column")
                    minval = col.minval
                    maxval = col.maxval + 1
                    norm_range = np.linspace(0.0, 1.0, count, dtype=np.float32)
                    prange = minval + (maxval - minval) * norm_range
                    for i in range(len(prange)-1):
                        val0 = np.ceil(prange[i])
                        val1 = np.ceil(prange[i+1])-1
                        assert val1 >= val0, (val0, val1)
                        query = new_query(table, ncols=1)
                        query.predicates[col.name] = ('[]', (val0, val1))
                        card, _ = oracle.query(query)
                        #  vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                        #  vec.append(card/table.row_num)

                        vec = [0.0, 1.0] * table.col_num
                        vec.append(card/table.row_num)
                        vec[col_id*2] = norm_range[i]
                        vec[col_id*2+1] = norm_range[i+1]
                        writer.writerow(vec)
                        L.info(f"# {i}: {query.predicates[col.name]}, card={card}\n\t{vec}")
            else:
                L.info("Real-value column")
                norm_range = np.linspace(0.0, 1.0, count, dtype=np.float32)
                prange = col.minval + (col.maxval - col.minval) * norm_range
                for i in range(len(prange)-1):
                    query = new_query(table, ncols=1)
                    query.predicates[col.name] = ('[]', (prange[i], prange[i+1]))
                    card, _ = oracle.query(query)
                    #  vec = query_2_quicksel_vector(query, table, discrete_cols).tolist()
                    #  vec.append(card/table.row_num)
                    vec = [0.0, 1.0] * table.col_num
                    vec.append(card/table.row_num)
                    vec[col_id*2] = norm_range[i]
                    vec[col_id*2+1] = norm_range[i+1]
                    writer.writerow(vec)
                    L.info(f"# {i}: {query.predicates[col.name]}, card={card}\n\t{vec}")
