import time
import logging
from typing import Dict, Any, Tuple
import numpy as np
from ..estimator import Estimator
from ..utils import run_test, evaluate
from ...constants import DATA_ROOT, MODEL_ROOT, NUM_THREADS, VALID_NUM_DATA_DRIVEN
from ...dataset.dataset import load_table
from ...workload.workload import load_queryset, load_labels, query_2_sql

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from ensemble_compilation.graph_representation import SchemaGraph, Table, QueryType
from ensemble_compilation.spn_ensemble import SPNEnsemble, read_ensemble
from data_preparation.join_data_preparation import JoinDataPreparator
from data_preparation.prepare_single_tables import prepare_all_tables
from aqp_spn.aqp_spn import AQPSPN
from aqp_spn.aqp_leaves import Sum, Categorical, IdentityNumericLeaf
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.structure.Base import get_nodes_by_type, Product
from evaluation.utils import parse_query

L = logging.getLogger(__name__)

class Args:
    def __init__(self, **kwargs):
        self.max_rows_per_hdf_file = 20000000
        self.hdf_sample_size = 1000000
        self.rdc_threshold = 0.3
        self.ratio_min_instance_slice = 0.01

        # overwrite parameters from user
        self.__dict__.update(kwargs)

def construct_schema(table):
    # construct a schema that has one table only
    csv_file = DATA_ROOT / table.dataset / f"{table.version}.csv"
    schema = SchemaGraph()
    schema.add_table(Table(f'"{table.name}"', # use table name in postgres since deepdb deal with sql directly
                           attributes=table.data.columns.values.tolist(),
                           csv_file_location=csv_file,
                           table_size=table.row_num))
    return schema

def get_deepdb_size(spn_ensemble):
    # only deal with single table, only have one spn
    spn = spn_ensemble.spns[0].mspn
    size = 0
    nodes = get_nodes_by_type(spn, Product)
    for node in nodes:
        size += len(node.children) + len(node.scope)

    nodes = get_nodes_by_type(spn, Sum)
    for node in nodes:
        assert len(node.children) == len(node.weights) == len(node.cluster_centers)
        assert len(node.cluster_centers[0]) == len(node.scope)
        num_child = len(node.children)
        num_var = len(node.scope)
        size += 2*num_child + num_var + num_var*num_child # children, weights, scope, cluster_centers

    nodes = get_nodes_by_type(spn, Categorical)
    for node in nodes:
        assert len(node.scope) == 1
        size += 2 + len(node.p) # scope, cardinality, p

    nodes = get_nodes_by_type(spn, IdentityNumericLeaf)
    for node in nodes:
        assert len(node.scope) == 1
        assert len(node.unique_vals) + 1 == len(node.prob_sum)
        size += 3 + len(node.unique_vals) + len(node.prob_sum) # scope, cardinality, null_value_prob, uniqe_vals, prob_sum

    # assume use 4 bytes to store all integers and floats
    return size * 4 / 1024 / 1024 #MB

def train_deepdb(seed, dataset, version, workload, params, sizelimit):
    L.info(f"params: {params}")
    args = Args(**params)

    # for sampling
    np.random.seed(seed)

    table = load_table(dataset, version)
    # load validation queries and labels
    valid_queries = load_queryset(dataset, workload)['valid'][:VALID_NUM_DATA_DRIVEN]
    labels = load_labels(dataset, version, workload)['valid'][:VALID_NUM_DATA_DRIVEN]

    schema = construct_schema(table)

    # convert data from csv to hdf
    hdf_path = DATA_ROOT / dataset / 'deepdb' / f"hdf-{version}"
    if hdf_path.is_dir():
        L.info('Use existing hdf file!')
    else:
        hdf_path.mkdir(parents=True)
        prepare_all_tables(schema, str(hdf_path), csv_seperator=',', max_table_data=args.max_rows_per_hdf_file)

    # generate SPN for table
    prep = JoinDataPreparator(hdf_path / 'meta_data.pkl', schema, max_table_data=args.max_rows_per_hdf_file)
    spn_ensemble = SPNEnsemble(schema)
    table_obj = schema.tables[0]
    L.info(f"table name: {table_obj.table_name}")
    df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(args.hdf_sample_size,
                                                                                 single_table=table_obj.table_name,
                                                                                 post_sampling_factor=1.0)
    assert len(df_samples) == min(args.hdf_sample_size, table.row_num), '{} != min({}, {})'.format(len(df_samples), args.hdf_sample_size, table.row_num)


    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-spn_sample{len(df_samples)}_rdc{args.rdc_threshold}_ms{args.ratio_min_instance_slice}-{seed}.pkl"

    # learn spn
    L.info(f"Start learning SPN for {table_obj.table_name}.")
    start_stmp = time.time()
    aqp_spn = AQPSPN(meta_types, null_values, full_join_est, schema, relationship_list=None,
                     full_sample_size=len(df_samples), table_set={table_obj.table_name},
                     column_names=list(df_samples.columns), table_meta_data=prep.table_meta_data)
    min_instance_slice = args.ratio_min_instance_slice * len(df_samples)
    aqp_spn.learn(df_samples.values, min_instances_slice=min_instance_slice, bloom_filters=False,
                  rdc_threshold=args.rdc_threshold)
    spn_ensemble.add_spn(aqp_spn)
    dur_min = (time.time() - start_stmp) / 60

    mb = get_deepdb_size(spn_ensemble)
    L.info(f"SPN built finished, time spent since start: {dur_min:.1f} mins with {mb:.2f}MB size of memory")
    L.info(f'Final SPN: {get_structure_stats_dict(spn_ensemble.spns[0].mspn)}')

    if sizelimit > 0 and mb > (sizelimit * table.data_size_mb):
        L.info(f"Exceeds size limit {mb:.2f}MB > {sizelimit} x {table.data_size_mb}, do not conintue!")
        return

    L.info(f"Evaluating on valid set with {VALID_NUM_DATA_DRIVEN} queries...")
    estimator = DeepDB(spn_ensemble, table, schema, 'valid')
    preds = []
    for q in valid_queries:
        est_card, _ = estimator.query(q)
        preds.append(est_card)
    _, metrics = evaluate(preds, [l.cardinality for l in labels])

    spn_ensemble.state = {
        'train_time': dur_min,
        'model_size': mb,
        'args': args,
        'device': 'cpu',
        'threads': NUM_THREADS,
        'dataset': table.dataset,
        'version': table.version,
        'valid_error': {workload: metrics}
    }

    # save spn to file
    spn_ensemble.save(model_file)
    L.info(f'Training finished! Save model to {model_file} Time spent since start: {dur_min:.2f} mins')

class DeepDB(Estimator):
    def __init__(self, spn_ensemble, table, schema, model_name):
        super(DeepDB, self).__init__(table=table, model=model_name)
        self.spn_ensemble = spn_ensemble
        self.schema = schema

    def query(self, query):
        sql = query_2_sql(query, self.table, aggregate=True, split=True)
        #  print(sql)
        query = parse_query(sql.strip(), self.schema)
        assert query.query_type == QueryType.CARDINALITY

        start_stmp = time.time()
        formula, factors, card, factor_values = self.spn_ensemble.cardinality(query, return_factor_values=True)
        dur_ms = (time.time() - start_stmp) * 1e3
        #  print(factors)
        #  print(factor_values)
        #  print(formula)
        return np.round(card), dur_ms

def load_deepdb(dataset: str, model_name: str) -> Tuple[Estimator, Dict[str, Any]]:
    model_file = MODEL_ROOT / dataset /f"{model_name}.pkl"
    L.info(f"load model from {model_file} ...")
    spn_ensemble = read_ensemble(model_file, build_reverse_dict=True)
    L.info(f'Get SPN: {get_structure_stats_dict(spn_ensemble.spns[0].mspn)}')

    state = spn_ensemble.state
    table = load_table(state['dataset'], state['version'])
    schema = construct_schema(table)
    estimator = DeepDB(spn_ensemble, table, schema, model_name)
    return estimator, state

def test_deepdb(dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        model: model file name
    """

    model_file = MODEL_ROOT / dataset /f"{params['model']}.pkl"
    L.info(f"load model from {model_file} ...")
    spn_ensemble = read_ensemble(model_file, build_reverse_dict=True)
    L.info(f'Get SPN: {get_structure_stats_dict(spn_ensemble.spns[0].mspn)}')

    state = spn_ensemble.state
    table = load_table(state['dataset'], state['version'])
    schema = construct_schema(table)
    estimator = DeepDB(spn_ensemble, table, schema, params['model'])

    run_test(dataset, version, workload, estimator, overwrite)

def update_deepdb(seed: int, dataset: str, new_version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    # for sampling
    np.random.seed(seed)
    # load old model
    new_table = load_table(dataset, new_version)
    model_path = MODEL_ROOT / new_table.dataset
    model_file = model_path /f"{params['model']}.pkl"
    L.info(f"load model from {model_file} ...")
    estimator, state = load_deepdb(dataset, params['model'])
    spn_ensemble = estimator.spn_ensemble

    old_version = state['version']
    args = state['args']
    old_table = load_table(dataset, old_version)
    # load updated data and save to csv
    updated_dataset = load_table(dataset, new_version)
    updated_dataset.data = updated_dataset.data.iloc[len(old_table.data):].sample(frac=0.01)
    updated_dataset.data.reset_index(drop=True)
    updated_dataset.version += '_cut'
    updated_dataset.name += '_cut'
    updated_dataset.data.to_csv(DATA_ROOT / dataset / f"{updated_dataset.version}.csv", index=False)
    updated_dataset.row_num = len(updated_dataset.data)
    
    L.info(f"Updated size {updated_dataset.row_num}")

    # load validation queries and labels
    valid_queries = load_queryset(dataset, workload)['valid'][:VALID_NUM_DATA_DRIVEN]
    labels = load_labels(dataset, new_version, workload)['valid'][:VALID_NUM_DATA_DRIVEN]

    schema = construct_schema(updated_dataset)
    L.info(f"{schema}")
    # convert data from csv to hdf
    hdf_path = DATA_ROOT / dataset / 'deepdb' / f"hdf-{updated_dataset.version}"
    if hdf_path.is_dir():
        L.info('Use existing hdf file!')
    else:
        hdf_path.mkdir(parents=True)
    prepare_all_tables(schema, str(hdf_path), csv_seperator=',', max_table_data=args.max_rows_per_hdf_file)

    # generate SPN for table
    prep = JoinDataPreparator(hdf_path / 'meta_data.pkl', schema, max_table_data=args.max_rows_per_hdf_file)
    table_obj = schema.tables[0]
    L.info(f"table name: {table_obj.table_name}")
    L.info(f"table attributes: {schema.tables[0].attributes}")
    df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(args.hdf_sample_size,
                                                                                 single_table=table_obj.table_name,
                                                                                 post_sampling_factor=1.0)
    # assert len(df_samples) == min(args.hdf_sample_size, old_table.row_num), '{} != min({}, {})'.format(len(df_samples), args.hdf_sample_size, old_table.row_num)

    
    # Update model
    L.info(f"Start learning SPN for {table_obj.table_name}.")
    start_stmp = time.time()
    spn_ensemble.spns[0].learn_incremental(df_samples.to_numpy())
    dur_min = (time.time() - start_stmp) / 60

    L.info(f"SPN update finished, time spent since start: {dur_min:.4f} mins")
    L.info(f'Final SPN: {get_structure_stats_dict(spn_ensemble.spns[0].mspn)}')

    # L.info(f"Evaluating on valid set with {VALID_NUM_DATA_DRIVEN} queries...")
    # estimator = DeepDB(spn_ensemble, new_table, schema, 'valid')
    # preds = []
    # for q in valid_queries:
    #     est_card, _ = estimator.query(q)
    #     preds.append(est_card)
    # _, metrics = evaluate(preds, [l.cardinality for l in labels])

    spn_ensemble.state['update_time'] = dur_min
    args = state['args']
    # save spn to file
    sample_size = min(args.hdf_sample_size, old_table.row_num)
    new_model_file = model_path / f"{new_table.version}-spn_sample{sample_size}_rdc{args.rdc_threshold}_ms{args.ratio_min_instance_slice}-{seed}.pkl"

    spn_ensemble.save(new_model_file)
    L.info(f'Updating finished! Save model to {new_model_file} Time spent since start: {dur_min:.4f} mins')


