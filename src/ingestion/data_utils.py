import streamlit as st 
import ingestion.query as qu
import json

def load_data(sql_path, table_path, use_small=False):
    sql_data = []
    table_data = {}
    st.write("Loading data from %s" % sql_path)
    with open(sql_path) as lines:
        for idx, line in enumerate(lines):
            if use_small and idx >= 1000:
                break
            sql = json.loads(line.strip())
            sql_data.append(sql)
    with open(table_path) as lines:
        for line in lines:
            tab = json.loads(line.strip())
            table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data
    return sql_data, table_data

def load_dataset(use_small=False):
    sql_data, table_data = load_data('data/preprocessed/train.jsonl', 
        'data/preprocessed/train.tables.jsonl', use_small=use_small)
    val_sql_data, val_table_data = load_data('data/preprocessed/dev.jsonl', 
        'data/preprocessed/dev.tables.jsonl', use_small=use_small)
    test_sql_data, test_table_data = load_data('data/preprocessed/test.jsonl', 
        'data/preprocessed/test.tables.jsonl', use_small=use_small)
    TRAIN_DB = 'data/preprocessed/train.db'
    DEV_DB = 'data/preprocessed/dev.db'
    TEST_DB = 'data/preprocessed/test.db'
    
    return sql_data, table_data, val_sql_data, val_table_data, \
        test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB

def print_sample_data(index, sql_data, table_data):
    query = qu.Query(sql_data[index]['sql']['sel'], sql_data[index]['sql']['agg'], 
        sql_data[index]['sql']['conds'])
    st.write('**Sample data:**')
    st.write('*Question*: %s' % sql_data[index][u'question'])
    st.write('*Query*: %s' % repr(query))
    st.write('*Table columns*: %s' % ', '.join(['{}: {}'.format(i, x) for i,x in \
        enumerate(table_data[sql_data[index][u'table_id']][u'header'])]))
