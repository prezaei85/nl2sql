#!/usr/bin/env python 
from __future__ import division
from builtins import bytes
import os
import argparse
import math
import codecs
import torch

from table.Translator import Translator
import table.IO
import opts
from itertools import takewhile, count
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
from path import Path
import glob
import json
from tqdm import tqdm
from lib.dbengine import DBEngine
from lib.query import Query


def main(anno_file_name, col_headers, raw_args=None):
    parser = argparse.ArgumentParser(description='evaluate.py')
    opts.translate_opts(parser)
    opt = parser.parse_args(raw_args)
    torch.cuda.set_device(opt.gpu)
    opt.db_file = os.path.join(opt.data_path, '{}.db'.format(opt.split))
    opt.pre_word_vecs = os.path.join(opt.data_path, 'embedding')
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    opts.train_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opt.anno = anno_file_name

    engine = DBEngine(opt.db_file)

    js_list = table.IO.read_anno_json(opt.anno)

    prev_best = (None, None)
    sql_query = []
    for fn_model in glob.glob(opt.model_path):

        opt.model = fn_model

        translator = Translator(opt, dummy_opt.__dict__)
        data = table.IO.TableDataset(js_list, translator.fields, None, False)
        test_data = table.IO.OrderedIterator(
            dataset=data, device=opt.gpu, batch_size=opt.batch_size, train=False, sort=True, sort_within_batch=False)

        # inference
        r_list = []
        for batch in test_data:
            r_list += translator.translate(batch)
        r_list.sort(key=lambda x: x.idx)
        pred = r_list[-1]
        sql_pred = {'agg':pred.agg, 'sel':pred.sel, 'conds': pred.recover_cond_to_gloss(js_list[-1])}
        sql_query = Query(sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
        try:
            ans_pred = engine.execute_query(
                js_list[-1]['table_id'], Query.from_dict(sql_pred), lower=True)
        except Exception as e:
            ans_pred = None
    return sql_query.get_complete_query(col_headers), ans_pred


