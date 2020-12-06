import os
import copy
import logging
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

from ..constants import DATA_ROOT, PKL_PROTO
from ..dtypes import is_categorical

L = logging.getLogger(__name__)

class Column(object):
    def __init__(self, name, data):
        self.name = name
        self.dtype = data.dtype

        # parse vocabulary
        self.vocab, self.has_nan = self.__parse_vocab(data)
        self.vocab_size = len(self.vocab)
        self.minval = self.vocab[1] if self.has_nan else self.vocab[0]
        self.maxval = self.vocab[-1]

    def __repr__(self):
        return f'Column({self.name}, type={self.dtype}, vocab size={self.vocab_size}, min={self.minval}, max={self.maxval}, has NaN={self.has_nan})'

    def __parse_vocab(self, data):
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        is_nan = pd.isnull(data)
        contains_nan = np.any(is_nan)
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        vs = np.sort(np.unique(data[~is_nan]))
        if contains_nan:
            vs = np.insert(vs, 0, np.nan)
        return vs, contains_nan

    def discretize(self, data):
        """Transforms data values into integers using a Column's vocabulary"""

        # pd.Categorical() does not allow categories be passed in an array
        # containing np.nan.  It makes it a special case to return code -1
        # for NaN values.
        if self.has_nan:
            bin_ids = pd.Categorical(data, categories=self.vocab[1:]).codes
            # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
            # add 1 to everybody
            bin_ids = bin_ids + 1
        else:
            # This column has no nan or nat values.
            bin_ids = pd.Categorical(data, categories=self.vocab).codes

        bin_ids = bin_ids.astype(np.int32, copy=False)
        assert (bin_ids >= 0).all(), (self, data, bin_ids)
        return bin_ids

    def normalize(self, data):
        """Normalize data to range [0, 1]"""
        minval = self.minval
        maxval = self.maxval
        # if column is not numerical, use descretized value
        if is_categorical(self.dtype):
            data = self.discretize(data)
            minval = 0
            maxval = self.vocab_size - 1
        data = np.array(data, dtype=np.float32)
        if minval >= maxval:
            L.warning(f"column {self.name} has min value {minval} >= max value{maxval}")
            return np.zeros(len(data)).astype(np.float32)
        val_norm = (data - minval) / (maxval - minval)
        return val_norm.astype(np.float32)

class Table(object):
    def __init__(self, dataset, version):
        self.dataset = dataset
        self.version = version
        self.name = f"{self.dataset}_{self.version}"
        L.info(f"start building data {self.name}...")

        # load data
        self.data = pd.read_pickle(DATA_ROOT / self.dataset / f"{self.version}.pkl")
        self.data_size_mb = self.data.values.nbytes / 1024 / 1024
        self.row_num = self.data.shape[0]
        self.col_num = len(self.data.columns)

        # parse columns
        self.parse_columns()
        L.info(f"build finished: {self}")
    
    def parse_columns(self):
        self.columns = OrderedDict([(col, Column(col, self.data[col])) for col in self.data.columns])

    def __repr__(self):
        return f"Table {self.name} ({self.row_num} rows, {self.data_size_mb:.2f}MB, columns:\n{os.linesep.join([repr(c) for c in self.columns.values()])})"

    def get_minmax_dict(self):
        minmax_dict = {}
        for i, col in enumerate(self.columns.values()):
            minmax_dict[i] = (col.minval, col.maxval)
        return minmax_dict

    def normalize(self, scale=1):
        data = copy.deepcopy(self.data)
        for cname, col in self.columns.items():
            data[cname] = col.normalize(data[cname].values) * scale
        return data

    def digitalize(self):
        data = copy.deepcopy(self.data)
        for cname, col in self.columns.items():
            if is_categorical(col.dtype):
                data[cname] = col.discretize(data[cname])
            elif col.has_nan:
                data[cname].fillna(0, inplace=True)
        return data

    def get_max_muteinfo_order(self):
        order = []

        # find the first column with maximum entropy
        max_entropy = float('-inf')
        first_col = None
        for c in self.columns.keys():
            e = entropy(self.data[c].value_counts())
            if e > max_entropy:
                first_col = c
                max_entropy = e
        assert first_col is not None, (first_col, max_entropy)
        order.append(first_col)
        sep = '|'
        chosen_data = self.data[first_col].astype(str) + sep

        # add the rest columns one by one by choosing the max mutual information with existing columns
        while len(order) < self.col_num:
            max_muinfo = float('-inf')
            next_col = None
            for c in self.columns.keys():
                if c in order: continue
                m = mutual_info_score(chosen_data, self.data[c])
                if m > max_muinfo:
                    next_col = c
                    max_muinfo = m
            assert next_col is not None, (next_col, max_entropy)
            order.append(next_col)
            # concate new chosen columns
            chosen_data = chosen_data + sep + self.data[next_col].astype(str)

        return order, [self.data.columns.get_loc(c) for c in order]

    def get_muteinfo(self, digital_data=None):
        data = digital_data if digital_data is not None else self.digitalize()
        muteinfo_dict = {}
        for c1 in self.columns.keys():
            muteinfo_dict[c1] = {}
            for c2 in self.columns.keys():
                if c1 != c2 and c2 in muteinfo_dict:
                    assert c1 in muteinfo_dict[c2], muteinfo_dict.keys()
                    muteinfo_dict[c1][c2] = muteinfo_dict[c2][c1]
                else:
                    muteinfo_dict[c1][c2] = mutual_info_score(data[c1], data[c2])
        return pd.DataFrame().from_dict(muteinfo_dict)

def dump_table(table: Table) -> None:
    with open(DATA_ROOT / table.dataset / f"{table.version}.table.pkl", 'wb') as f:
        pickle.dump(table, f, protocol=PKL_PROTO)

def load_table(dataset: str, version: str, overwrite: bool=False) -> Table:
    table_path = DATA_ROOT / dataset / f"{version}.table.pkl"

    if not overwrite and table_path.is_file():
        L.info("table exists, load...")
        with open(table_path, 'rb') as f:
            table = pickle.load(f)
        L.info(f"load finished: {table}")
        return table

    table = Table(dataset, version)
    L.info("dump table to disk...")
    dump_table(table)
    return table

def dump_table_to_num(dataset: str, version: str) -> None:
    table = load_table(dataset, version)
    num_data = table.digitalize()
    csv_path = DATA_ROOT / dataset / f"{version}_num.csv"
    L.info(f"dump csv file to {csv_path}")
    num_data.to_csv(csv_path, index=False)


if __name__ == '__main__':
    #  table = load_table('forest')
    #  print(table.get_max_muteinfo_order())
    # 7 1 8 6 5 9 0 4 3 2

    #  table = load_table('census')
    #  print(table.get_max_muteinfo_order())
    # 4 3 2 0 6 12 7 5 1 13 9 10 8 11

    table = Table('census', 'original')
    print(table)
    #  print(table.get_max_muteinfo_order())
    # 4 0 1 2 3 5 8 7 6
