import sys
import os
from parse import *
from datetime import datetime as dt

# path hack
sys.path.append(os.getcwd())
sys.path.append('..')

TIME_FMT = '%Y-%m-%d %H:%M:%S,%f'

def get_gen_query_time(logfile, training_size):
    '''return time_gen_train_query, time_gen_train_label'''
    t_tr_query = [[],[]]
    t_tr_label = [[],[]]
    # time_update_model = [[],[]]
    with open(logfile, 'r') as log_f:
        lines = log_f.readlines()
        for line in lines:
            line = line.strip()
            # parse time for training query update
            s_tr_query=parse("[{time} INFO] lecarb.workload.gen_workload: Start generate workload with {train_num:d} queries for train...", line)
            e_tr_query=parse("[{time} INFO] lecarb.workload.gen_workload: Start generate workload with {test_num:d} queries for valid...", line)
            if s_tr_query and s_tr_query['train_num'] == training_size:
                t_tr_query[0].append(dt.strptime(s_tr_query['time'], TIME_FMT))
            if e_tr_query:
                t_tr_query[1].append(dt.strptime(e_tr_query['time'], TIME_FMT))
            
            # parse time for training label update
            s_tr_label=parse("[{time} INFO] lecarb.workload.gen_label: Updating ground truth labels for the workload, with sample size {}...", line)
            e_tr_label=parse("[{time} INFO] lecarb.workload.gen_label: Dump labels to disk...", line)
            if s_tr_label:
                t_tr_label[0].append(dt.strptime(s_tr_label['time'], TIME_FMT))
            if e_tr_label:
                t_tr_label[1].append(dt.strptime(e_tr_label['time'], TIME_FMT))
        # print(t_tr_query, t_tr_label)

        time_gen_tr_query = 0
        time_gen_tr_label = 0
        if len(t_tr_query[0]) >= 1:
            time_gen_tr_query = (t_tr_query[1][0] - t_tr_query[0][0]).total_seconds()
        if len(t_tr_label[0]) >= 1:
            time_gen_tr_label = (t_tr_label[1][0] - t_tr_label[0][0]).total_seconds()
    return time_gen_tr_query, time_gen_tr_label

def get_lw_nn_training_time(logfile):
    with open(logfile, 'r') as log_f:
            lines = log_f.readlines()
            for line in lines:
                line = line.strip()
                update_time=parse("[{} INFO] lecarb.estimator.lw.lw_nn: Training finished! Time spent since start: {train_time:f} mins", line)
                if update_time:
                    return update_time['train_time'] 
    return 0

def get_postgres_time(logfile):
    with open(logfile, 'r') as logf:
        lines = logf.readlines()
        for line in lines:
            line = line.strip()
            # parse time for training query update
            update_time=parse("[{} INFO] lecarb.estimator.postgres: construct statistics finished, using {update_time:f} minutes, All statistics consumes {} MBs", line)
            if update_time:
                return update_time['update_time']
    return 0
        
def get_mysql_time(logfile):
    with open(logfile, 'r') as logf:
        lines = logf.readlines()
        for line in lines:
            line = line.strip()
            # parse time for training query update
            update_time=parse("[{} INFO] lecarb.estimator.mysql: construct statistics finished, using {update_time:f} minutes", line)
            if update_time:
                return update_time['update_time']
    return 0
