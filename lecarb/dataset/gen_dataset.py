import random
import logging
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, truncexpon, genpareto
from typing import Dict, Any

from .dataset import load_table
from ..constants import DATA_ROOT

L = logging.getLogger(__name__)

def get_truncated_normal(mean=0, sd=100, low=0, upp=1000):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_truncated_expon(scale=100, low=0, upp=1000):
    return truncexpon(b=(upp-low)/scale, loc=low, scale=scale)

def generate_dataset(
    seed: int, dataset: str, version: str,
    params: Dict[str, Any], overwrite: bool
) -> None:
    path = DATA_ROOT / dataset
    path.mkdir(exist_ok=True)
    csv_path = path / f"{version}.csv"
    pkl_path = path / f"{version}.pkl"
    if not overwrite and csv_path.is_file():
        L.info(f"Dataset path exists, do not continue")
        return

    row_num = params['row_num']
    col_num = params['col_num']
    dom = params['dom']
    corr = params['corr']
    skew = params['skew']

    if col_num != 2:
        L.info("For now only support col=2!")
        exit(0)

    L.info(f"Start generate dataset with {col_num} columns and {row_num} rows using seed {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # generate the first column according to skew
    col0 = np.arange(dom) # make sure every domain value has at least 1 value
    tmp = genpareto.rvs(skew-1, size=row_num-len(col0)) # c = skew - 1, so we can have c >= 0
    tmp = ((tmp - tmp.min()) / (tmp.max() - tmp.min())) * dom # rescale generated data to the range of domain
    col0 = np.concatenate((col0, np.clip(tmp.astype(int), 0, dom-1)))

    # generate the second column according to the first
    col1 = []
    for c0 in col0:
        col1.append(c0 if np.random.uniform(0, 1) <= corr else np.random.choice(dom))

    df = pd.DataFrame(data={'col0': col0, 'col1': col1})

    L.info(f"Dump dataset {dataset} as version {version} to disk")
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    load_table(dataset, version)
    L.info(f"Finish!")

