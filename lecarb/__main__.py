"""Le Carb - LEarned CARdinality estimator Benchmark

Usage:
  lecarb workload gen [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [--params <params>] [--no-label] [-o <old_version>] [-r <ratio>]
  lecarb workload label [-d <dataset>] [-v <version>] [-w <workload>]
  lecarb workload update-label [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [--sample-ratio <sample_size>]
  lecarb workload merge [-d <dataset>] [-v <version>] [-w <workload>]
  lecarb workload dump [-d <dataset>] [-v <version>] [-w <workload>]
  lecarb workload quicksel [-d <dataset>] [-v <version>] [-w <workload>] [--params <params>] [--overwrite]
  lecarb dataset table [-d <dataset>] [-v <version>] [--overwrite]
  lecarb dataset gen [-s <seed>] [-d <dataset>] [-v <version>] [--params <params>] [--overwrite]
  lecarb dataset update [-s <seed>] [-d <dataset>] [-v <version>] [--params <params>] [--overwrite]
  lecarb dataset dump [-s <seed>] [-d <dataset>] [-v <version>]
  lecarb train [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [-e <estimator>] [--params <params>] [--sizelimit <sizelimit>]
  lecarb test [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [-e <estimator>] [--params <params>] [--overwrite]
  lecarb report [-d <dataset>] [--params <params>]
  lecarb report-dynamic [-d <dataset>] [--params <params>]
  lecarb update-train [-s <seed>] [-d <dataset>] [-v <version>] [-w <workload>] [-e <estimator>] [--params <params>] [--overwrite]

Options:
  -s, --seed <seed>                Random seed.
  -d, --dataset <dataset>          The input dataset [default: census13].
  -v, --dataset-version <version>  Dataset version [default: original].
  -w, --workload <workload>        Name of the workload [default: base].
  -e, --estimator <estimator>      Name of the estimator [default: naru].
  --params <params>                Parameters that are needed.
  --sizelimit <sizelimit>          Size budget of method, percentage to data size [default: 0.015].
  --no-label                       Do not generate ground truth label when generate workload.
  --overwrite                      Overwrite the result.
  -o, --old-version <old_version>  When data updates, query should focus more on the new data. The <old version> is what QueryGenerator refers to.
  -r, --win-ratio <ratio>          QueryGen only touch last <win_ratio> * size_of(<old version>).
  --sample-ratio <sample-ratio>    Update query set with sample of dataset
  -h, --help                       Show this screen.
"""
from ast import literal_eval
from time import time

from docopt import docopt

from .workload.gen_workload import generate_workload
from .workload.gen_label import generate_labels, update_labels
from .workload.merge_workload import merge_workload
from .workload.dump_quicksel import dump_quicksel_query_files, generate_quicksel_permanent_assertions
from .dataset.dataset import load_table, dump_table_to_num
from .dataset.gen_dataset import generate_dataset
from .dataset.manipulate_dataset import gen_appended_dataset
from .estimator.sample import test_sample
from .estimator.postgres import test_postgres
from .estimator.mysql import test_mysql
from .estimator.mhist import test_mhist
from .estimator.bayesnet import test_bayesnet
from .estimator.feedback_kde import test_kde
from .estimator.utils import report_errors, report_dynamic_errors
from .estimator.naru.naru import train_naru, test_naru, update_naru
from .estimator.mscn.mscn import train_mscn, test_mscn
from .estimator.lw.lw_nn import train_lw_nn, test_lw_nn
from .estimator.lw.lw_tree import train_lw_tree, test_lw_tree
from .estimator.deepdb.deepdb import train_deepdb, test_deepdb, update_deepdb
from .workload.workload import dump_sqls

if __name__ == "__main__":
    args = docopt(__doc__, version="Le Carb 0.1")

    seed = args["--seed"]
    if seed is None:
        seed = int(time())
    else:
        seed = int(seed)

    if args["workload"]:
        if args["gen"]:
            generate_workload(
                seed,
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                name=args["--workload"],
                no_label = args["--no-label"],
                old_version=args["--old-version"],
                win_ratio=args["--win-ratio"],
                params = literal_eval(args["--params"])
            )
        elif args["label"]:
            generate_labels(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"]
            )
        elif args["update-label"]:
            update_labels(
                seed,
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"],
                sampling_ratio=literal_eval(args["--sample-ratio"])
            )
        elif args["merge"]:
            merge_workload(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"]
            )
        elif args["quicksel"]:
            dump_quicksel_query_files(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"],
                overwrite=args["--overwrite"]
            )
            generate_quicksel_permanent_assertions(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                params=literal_eval(args["--params"]),
                overwrite=args["--overwrite"]
            )
        elif args["dump"]:
            dump_sqls(
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                workload=args["--workload"])
        else:
            raise NotImplementedError
        exit(0)

    if args["dataset"]:
        if args["table"]:
            load_table(args["--dataset"], args["--dataset-version"], overwrite=args["--overwrite"])
        elif args["gen"]:
            generate_dataset(
                seed,
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                params=literal_eval(args["--params"]),
                overwrite=args["--overwrite"]
            )
        elif args["update"]:
            gen_appended_dataset(
                seed,
                dataset=args["--dataset"],
                version=args["--dataset-version"],
                params=literal_eval(args["--params"]),
                overwrite=args["--overwrite"]
            )
        elif args["dump"]:
            dump_table_to_num(args["--dataset"], args["--dataset-version"])
        else:
            raise NotImplementedError
        exit(0)

    if args["train"]:
        dataset = args["--dataset"]
        version = args["--dataset-version"]
        workload = args["--workload"]
        params = literal_eval(args["--params"])
        sizelimit = float(args["--sizelimit"])

        if args["--estimator"] == "naru":
            train_naru(seed, dataset, version, workload, params, sizelimit)
        elif args["--estimator"] == "mscn":
            train_mscn(seed, dataset, version, workload, params, sizelimit)
        elif args["--estimator"] == "deepdb":
            train_deepdb(seed, dataset, version ,workload, params, sizelimit)
        elif args["--estimator"] == "lw_nn":
            train_lw_nn(seed, dataset, version ,workload, params, sizelimit)
        elif args["--estimator"] == "lw_tree":
            train_lw_tree(seed, dataset, version ,workload, params, sizelimit)
        else:
            raise NotImplementedError
        exit(0)

    if args["test"]:
        dataset = args["--dataset"]
        version = args["--dataset-version"]
        workload = args["--workload"]
        params = literal_eval(args["--params"])
        overwrite = args["--overwrite"]

        if args["--estimator"] == "sample":
            test_sample(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "postgres":
            test_postgres(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "mysql":
            test_mysql(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "mhist":
            test_mhist(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "bayesnet":
            test_bayesnet(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "kde":
            test_kde(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "naru":
            test_naru(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "mscn":
            test_mscn(dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "deepdb":
            test_deepdb(dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "lw_nn":
            test_lw_nn(dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "lw_tree":
            test_lw_tree(dataset, version, workload, params, overwrite)
        else:
            raise NotImplementedError
        exit(0)

    if args["report"]:
        dataset = args["--dataset"]
        params = literal_eval(args["--params"])
        report_errors(dataset, params['file'])
        exit(0)
    
    if args["report-dynamic"]:
        dataset = args["--dataset"]
        params = literal_eval(args["--params"])
        report_dynamic_errors(dataset, params['old_new_file'], params['new_new_file'], params['T'], params['update_time'])
        exit(0)

    if args["update-train"]:
        dataset = args["--dataset"]
        version = args["--dataset-version"]
        workload = args["--workload"]
        params = literal_eval(args["--params"])
        overwrite = args["--overwrite"]

        if args["--estimator"] == "naru":
            update_naru(seed, dataset, version, workload, params, overwrite)
        elif args["--estimator"] == "deepdb":
            update_deepdb(seed, dataset, version, workload, params, overwrite)
        else:
            raise NotImplementedError
        exit(0)
