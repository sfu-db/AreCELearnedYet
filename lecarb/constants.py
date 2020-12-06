import os
from pathlib import Path
import torch

DATA_ROOT = Path(os.environ["DATA_ROOT"])
OUTPUT_ROOT = Path(os.environ["OUTPUT_ROOT"])
MODEL_ROOT = OUTPUT_ROOT / "model"
RESULT_ROOT = OUTPUT_ROOT / "result"
LOG_ROOT = OUTPUT_ROOT / "log"

DATABASE_URL = os.environ["DATABASE_URL"]
KDE_DATABASE_URL = os.environ["KDE_DATABASE_URL"]
MYSQL_HOST = os.environ["MYSQL_HOST"]
MYSQL_PORT = os.environ["MYSQL_PORT"]
MYSQL_DB = os.environ["MYSQL_DB"]
MYSQL_USER = os.environ["MYSQL_USER"]
MYSQL_PSWD = os.environ["MYSQL_PSWD"]

PKL_PROTO = 4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_THREADS = int(os.environ.get("CPU_NUM_THREADS", os.cpu_count()))

VALID_NUM_DATA_DRIVEN = 100
