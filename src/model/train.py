
import streamlit as st 
from ingestion.data_utils import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
        help='Batch size for training.')
    parser.add_argument('--use_gpu', action='store_true', 
        help='If set, use GPU.')
    parser.add_argument('--use_small', action='store_true', 
        help='If set, use small data; used for fast debugging.')
    args = parser.parse_args()

    sql_data, table_data, val_sql_data, val_table_data, \
        test_sql_data, test_table_data, \
        TRAIN_DB, DEV_DB, TEST_DB = load_dataset(args.use_small)

    print_sample_data(15, sql_data, table_data)



