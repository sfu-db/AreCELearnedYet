import random
import logging
import numpy as np
from typing import Dict, Any
import copy

from . import generator
from .generator import QueryGenerator
from .gen_label import generate_labels_for_queries
from .workload import dump_queryset, dump_labels
from ..dataset.dataset import load_table

L = logging.getLogger(__name__)

def get_focused_table(table, ref_table, win_ratio):
    focused_table = copy.deepcopy(table)
    win_size = int(win_ratio * len(ref_table.data))
    focused_table.data = focused_table.data.tail(win_size).reset_index(drop=True)
    focused_table.parse_columns()
    return focused_table

def generate_workload(
    seed: int, dataset: str, version: str,
    name: str, no_label: bool, old_version: str, win_ratio: str,
    params: Dict[str, Dict[str, Any]]
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    attr_funcs = {getattr(generator, f"asf_{a}"): v for a, v in params['attr'].items()}
    center_funcs = {getattr(generator, f"csf_{c}"): v for c, v in params['center'].items()}
    width_funcs = {getattr(generator, f"wsf_{w}"): v for w, v in params['width'].items()}

    L.info("Load table...")
    table = load_table(dataset, version)
    if old_version and win_ratio:
        L.info(f"According to {old_version}, generate queries for updated data in {version}...")
        win_ratio = float(win_ratio)
        assert 0<win_ratio<=1
        old_table = load_table(dataset, old_version)
        query_table = get_focused_table(table, old_table, win_ratio)
        qgen = QueryGenerator(
                table=query_table,
                attr=attr_funcs,
                center=center_funcs,
                width=width_funcs,
                attr_params=params.get('attr_params') or {},
                center_params=params.get('center_params') or {},
                width_params=params.get('width_params') or {})
    else:
        qgen = QueryGenerator(
            table=table,
            attr=attr_funcs,
            center=center_funcs,
            width=width_funcs,
            attr_params=params.get('attr_params') or {},
            center_params=params.get('center_params') or {},
            width_params=params.get('width_params') or {})

    queryset = {}
    for group, num in params['number'].items():
        L.info(f"Start generate workload with {num} queries for {group}...")
        queries = []
        for i in range(num):
            queries.append(qgen.generate())
            if (i+1) % 1000 == 0:
                L.info(f"{i+1} queries generated")
        queryset[group] = queries

    L.info("Dump queryset to disk...")
    dump_queryset(dataset, name, queryset)

    if no_label:
        L.info("Finish without generating corresponding ground truth labels")
        return

    L.info("Start generate ground truth labels for the workload...")
    labels = generate_labels_for_queries(table, queryset)

    L.info("Dump labels to disk...")
    dump_labels(dataset, version, name, labels)
