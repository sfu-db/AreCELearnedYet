import random
import logging
import pickle
import numpy as np
import math
import pandas as pd
from scipy.stats import truncnorm, truncexpon, genpareto
from typing import Dict, Any, Tuple
from copy import deepcopy

from .dataset import load_table
from ..constants import DATA_ROOT, PKL_PROTO

L = logging.getLogger(__name__)

# Independence data: Random by each column
def get_random_data(dataset: str, version: str, overwrite=False) -> Tuple[pd.DataFrame, str]:
    rand_version = f"{version}_ind"
    random_file = DATA_ROOT / dataset / f"{rand_version}.pkl"
    if not overwrite and random_file.is_file():
        L.info(f"Dataset path exists, using it")
        return pd.read_pickle(random_file), rand_version
    
    df = pd.read_pickle(DATA_ROOT / dataset / f"{version}.pkl")
    for col in df.columns:
        df[col] = df[col].sample(frac=1).reset_index(drop=True)
    pd.to_pickle(df, random_file, protocol=PKL_PROTO)
    return df, rand_version

# Max Spearman correlation data: sort by each column
def get_sorted_data(dataset: str, version: str, overwrite=False) -> Tuple[pd.DataFrame, str]:
    sort_version = f"{version}_cor"
    sorted_file = DATA_ROOT / dataset / f"{sort_version}.pkl"
    if not overwrite and sorted_file.is_file():
        return pd.read_pickle(sorted_file), sort_version
    
    df = pd.read_pickle(DATA_ROOT / dataset / f"{version}.pkl")
    for col in df.columns:
        df[col] = df[col].sort_values().reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    pd.to_pickle(df, sorted_file, protocol=PKL_PROTO)
    return df, sort_version

# Get skew data by tuple level frequent rank.
def get_skew_data(dataset: str = 'census', version: str = 'original', sample_ratio=0.0005, overwrite=False) -> Tuple[pd.DataFrame, str]:
    skew_version = f"{version}_skew"
    skew_file = DATA_ROOT / dataset / f"{skew_version}.pkl"
    if not overwrite and skew_file.is_file():
        return pd.read_pickle(skew_file), skew_version
    
    df = pd.read_pickle(DATA_ROOT / dataset / f"{version}.pkl")


    rank_df = pd.DataFrame(0.0, index=range(len(df)), columns=['rank_sum']).astype(np.float32)
    for col in df.columns:
        rank_df['rank_sum'] += df[col].map(df[col].value_counts().div(len(rank_df))).astype(np.float32)
        print(f"{col} frequency calculation finished!")
    selected_id = rank_df.sort_values(by='rank_sum').head(round(len(df)*sample_ratio)).index
    sk_df = df.iloc[selected_id]
    sk_df = pd.concat([sk_df] * int(1/sample_ratio + 1), ignore_index=True).head(len(df))
    pd.to_pickle(sk_df, skew_file, protocol=PKL_PROTO)
    return sk_df, skew_version



def append_data(dataset: str, version_target: str, version_from: str, interval=0.2):
    df_target = pd.read_pickle(DATA_ROOT / dataset / f"{version_target}.pkl")
    df_from = pd.read_pickle(DATA_ROOT / dataset / f"{version_from}.pkl")

    row_num = len(df_from)
    l = 0
    r = l + interval
    if r <= 1:
        L.info(f"Start appending {version_target} with {version_from} in [{l}, {r}]")
        df_target = df_target.append(df_from[int(l*row_num): int(r*row_num)], ignore_index=True, sort=False)
        pd.to_pickle(df_target, DATA_ROOT / dataset / f"{version_target}+{version_from}_{r:.1f}.pkl")
        df_target.to_csv(DATA_ROOT / dataset / f"{version_target}+{version_from}_{r:.1f}.csv", index=False)
        load_table(dataset, f"{version_target}+{version_from}_{r:.1f}")
    else:
        L.info(f"Appending Fail! Batch size is too big!")



def gen_appended_dataset(
    seed: int, dataset: str, version: str, 
    params: Dict[str, Any], overwrite: bool
    ) -> None:
    random.seed(seed)
    np.random.seed(seed)
    update_type = params.get('type')
    batch_ratio = params.get('batch_ratio')
    L.info(f"Start generating appended data for {dataset}/{version}")

    if update_type == 'ind':
        _, rand_version = get_random_data(dataset, version, overwrite=overwrite)
        append_data(dataset, version, rand_version, interval=batch_ratio)
    elif update_type == 'cor':
        _, sort_version = get_sorted_data(dataset, version, overwrite=overwrite)
        append_data(dataset, version, sort_version, interval=batch_ratio)
    elif update_type == 'skew':
        _, skew_version = get_skew_data(dataset, version,
                                        sample_ratio=float(params['skew_size']), overwrite=overwrite)
        append_data(dataset, version, skew_version, interval=batch_ratio)
    else:
        raise NotImplementedError
    L.info("Finish updating data!")


