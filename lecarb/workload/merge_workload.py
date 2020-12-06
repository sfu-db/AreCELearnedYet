import logging
from .workload import load_queryset, load_labels, dump_queryset, dump_labels

L = logging.getLogger(__name__)

def merge_workload(dataset: str, version: str, workload: str, count: int=10) -> None:
    queryset = {'train': [], 'valid': [], 'test': []}
    labels = {'train': [], 'valid': [], 'test': []}

    for i in range(count):
        L.info(f"Merge querset {workload}_{i}...")
        qs = load_queryset(dataset, f"{workload}_{i}")
        ls = load_labels(dataset, version, f"{workload}_{i}")
        for k in queryset.keys():
            #  print(f"{k}: {ls[k][0]}")
            queryset[k] += qs[k]
            labels[k] += ls[k]

    for k in queryset.keys():
        L.info(f"Final queryset has {len(queryset[k])} queries with {len(labels[k])} labels")

    L.info("Dump queryset and labels...")
    dump_queryset(dataset, workload, queryset)
    dump_labels(dataset, version, workload, labels)
    L.info(f"Done, run: rm data/{dataset}/workload/{workload}_[0-9]* to remove temporary files")
