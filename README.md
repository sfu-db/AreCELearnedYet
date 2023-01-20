# Are We Ready For Learned Cardinality Estimation?

**Our paper can be found at [arxiv](https://arxiv.org/abs/2012.06743) and [vldb](http://www.vldb.org/pvldb/vol14/p1640-wang.pdf).**

## Development Environment Setup

Setup:
* Install Just
  * MacOS: `brew install just`
  * Linux: `curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin`
* Install Poetry: `pip install poetry`
* Install Python dependencies: `just install-dependencies`

We define all the commands used in this work in `Justfile`. Run `just -l` for a list of supported tasks.

All the environmental configurations (e.g. data path, database configurations) are set in file `.env`.

## Dataset

Download the real-world datasets and workloads from [here](https://www.dropbox.com/s/5bmvc1si5hysapf/data.tar.gz?dl=0).

The path of the data is defined in `.env` as variable `DATA_ROOT`. We support dataset with different versions, typically a csv file is located at: `{DATA_ROOT}/{dataset name}/{version name}.csv`.

We define the `Table` object, which contains both data, some commonly used statistics and functions for convenient usage. Please refer to `lecarb/dataset/dataset.py` for details. (Most of the methods in our repo take `Table` as the dataset input.)

- Example: Given a csv file of census dataset (name: census13, version: original), generate the Table object
```bash
# 1. convert csv file to pickle
just csv2pkl data/census13/original.csv

# 2. convert to Table object
just pkl2table census13 original
```

- Example: Generate synthetic dataset with s=1.0, c=1.0, d=1000 (dataset name: dom1000, version: skew1.0_corr1.0)
```bash
just data-gen 1.0 1.0 1000
```
If we want to update the dataset, please run the command in the following format:

`just append-data-{update} {seed} {dataset} {version} {interval}`,

where {update} can be chosen from `cor` and `skew`. {seed} is the random seed. {dataset} is the dataset name. {version} is the version of the data. {interval} is between 0 and 1. It decides the ratio of the data to be appended. 



- Example: Generate appended the dataset (name: census13, version: original) with correlated (update: cor) data:
```
just append-data-cor 123 census13 original 0.2
```
The appended data will be located at: `{DATA_ROOT}/{dataset name}/{version}+{version}_{update}_{interval}.pkl`

## Workload
We adopt a unified workload generation framework to produce synthetic queries that we use in all the experiments. Specifically, in our framework each query is generated through three steps:

1. Choose a set of attributes to place predicates.
2. Select the query center for each predicate.
3. Determine the operator for each predicate (as well as widths for range predicates).

We have different implementations of each step in `lecarb/workload/generator.py` (function names start with `asf_`, `csf_` and `wsf_` respectively), user can also add customized implementations to the code for more variations.

- Example: generate workload used in static experiment for census dataset (workloads for real-world datasets used in the paper are already provided [here](https://www.dropbox.com/s/5bmvc1si5hysapf/data.tar.gz?dl=0))
```bash
# generate workload for small datasets (labels are generated in the same time)
just wkld-gen-base census13 original base

# for large datasets, start 10 processes to generate workload and then merge
just wkld-gen-mth10 census13 original base
just wkld-merge census13 original base
rm data/census13/workload/base_[0-9]*
```

- Example: generate workload for synthetic dataset (name: dom1000, version: skew1.0_corr1.0) used in the paper, check [hyper-params.md](./hyper-params.md#preparation) for a whole prepartion procedure (data generation and workload & label generation) of the micro-benchmark.
```base
# 1. generate workload (no labels generated)
just wkld-vood dom1000 skew1.0_corr1.0

# 2. generate labels
just wkld-label dom1000 skew1.0_corr1.0 vood
```

## Train & Test

Training and test commands for all the estimators are defined in `Justfile`, for hyper-parameters used and examples please refer to [hyper-params.md](./hyper-params.md).

Generated models are located at `{OUTPUT_ROOT}/model/{dataset name}/` and prediction results are at `{OUTPUT_ROOT}/result/{dataset name}/` in csv format.

Run `just report-error {output file name} {dataset name}` to see different error metrics of the **static** experiment result.

## Run dynamic experiments:

Dynamic experiment related code for reproducibility is in `dynamic-exp/` and commands can be found in `Justfile`.

(1) To run all dynmaic experiments, run `bash dynamic-exp/dynamic_exp.sh`. It includes all commands for dynamic experiment.

- Example: we want to run dynamic experiment for mscn on data 'census13'. We could run the following command:

```
just dynamic-mscn-census13 census13 original base cor 0.2 10000 123
```

'original' is the old version of census13. 'base' is the training workload generation method. 'cor' is the correlation change we consider for data update. '0.2' is the appended size of data (i.e. 20% of the 'original data'). '10000' is the size of training workload. '123' is the random seed.

(2) Run the following command to see different error metrics of the dynamic experiment errors.

`just report-error-dynamic {dataset} {stale model result file} {update model result file} {T} {model update time}`,

where {model update time} can be extracted through parsing the logging files. `dynamic-exp/parse_log_exmaple.py` provides some example scipts of extracting {model update time}.

For convinience usage, we put the hyperparameters of deferent models in `dynamic-exp/best_hp.py`. It copies the best hyperparameters we tested from [hyper-params.md](./hyper-params.md).

## Code References:

* Naru (including implementation of BayesNet): https://github.com/naru-project/naru
* MSCN: https://github.com/andreaskipf/learnedcardinalities
* DeepDB: https://github.com/DataManagementLab/deepdb-public
* QuickSel: https://github.com/illinoisdata/quicksel
* KDE-FB: https://github.com/martinkiefer/feedback-kde

Our forked repos:
* QuickSel:
  * Change: adding a new test class
  * Link: https://github.com/sfu-db/quicksel
* KDE-FB:
  * Making the code support tables with <=15 columns (original code has the limitation <=10)
  * Link: https://github.com/sfu-db/feedback-kde
