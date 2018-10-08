# This is a modified version of annotate.py to be used with external questions
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import records
# import ujson as json
import json
#from stanza.nlp.corenlp import CoreNLPClient
import copy
from lib.common import count_lines, detokenize
from lib.query import Query, agg_ops, cond_ops
import corenlp
import datetime
from shutil import copyfile
import spacy

client = None

def annotate(client, sentence, lower=True):
    words, gloss, after = [], [], []
    for s in client.annotate(sentence).sentence:
        for t in s.token:
            words.append(t.word)
            gloss.append(t.originalText)
            after.append(t.after)
    if lower:
        words = [w.lower() for w in words]
    return {
        'gloss': gloss,
        'words': words,
        'after': after,
    }


def annotate_example(client, example, table):
    ann = {'table_id': example['table_id']}
    ann['question'] = annotate(client, example['question'])
    ann['table'] = {
        'header': [annotate(client, h) for h in table['header']],
    }
    ann['query'] = sql = copy.deepcopy(example['sql'])
    for c in ann['query']['conds']:
        c[-1] = annotate(client, str(c[-1]))

    q1 = 'SYMSELECT SYMAGG {} SYMCOL {}'.format(
        agg_ops[sql['agg']], table['header'][sql['sel']])
    q2 = ['SYMCOL {} SYMOP {} SYMCOND {}'.format(
        table['header'][col], cond_ops[op], detokenize(cond)) for col, op, cond in sql['conds']]
    if q2:
        q2 = 'SYMWHERE ' + ' SYMAND '.join(q2) + ' SYMEND'
    else:
        q2 = 'SYMEND'
    inp = 'SYMSYMS {syms} SYMAGGOPS {aggops} SYMCONDOPS {condops} SYMTABLE {table} SYMQUESTION {question} SYMEND'.format(
        syms=' '.join(['SYM' + s for s in Query.syms]),
        table=' '.join(['SYMCOL ' + s for s in table['header']]),
        question=example['question'],
        aggops=' '.join([s for s in agg_ops]),
        condops=' '.join([s for s in cond_ops]),
    )
    ann['seq_input'] = annotate(client, inp)
    out = '{q1} {q2}'.format(q1=q1, q2=q2) if q2 else q1
    ann['seq_output'] = annotate(client, out)
    ann['where_output'] = annotate(client, q2)
    assert 'symend' in ann['seq_output']['words']
    assert 'symend' in ann['where_output']['words']
    return ann


def is_valid_example(e):
    if not all([h['words'] for h in e['table']['header']]):
        return False
    headers = [detokenize(h).lower() for h in e['table']['header']]
    if len(headers) != len(set(headers)):
        return False
    input_vocab = set(e['seq_input']['words'])
    for w in e['seq_output']['words']:
        if w not in input_vocab:
            print('query word "{}" is not in input vocabulary.\n{}'.format(
                w, e['seq_input']['words']))
            return False
    input_vocab = set(e['question']['words'])
    for col, op, cond in e['query']['conds']:
        for w in cond['words']:
            if w not in input_vocab:
                print('cond word "{}" is not in input vocabulary.\n{}'.format(
                    w, e['question']['words']))
                return False
    return True


def annotate_question(question, table_id, dir_in, dir_out, split):
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    ftable = os.path.join(dir_in, split) + '.tables.jsonl'
    fout = os.path.join(dir_out, split) + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.jsonl'
    fbase = os.path.join(dir_in, split) + '_base.jsonl'
    copyfile(fbase, fout)

    with open(ftable) as ft, open(fout, 'a') as fo, corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
        tables = {}
        for line in ft:
            d = json.loads(line)
            tables[d['id']] = d # to get table headers         
        raw_data = {"phase": 1, "table_id": table_id, "question": question, "sql": {"sel": 0, "conds": [[0, 0, 0]], "agg": 0}}
        a = annotate_example(client, raw_data, tables[table_id])
        # if not is_valid_example(a):
        #    raise Exception(str(a))
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(question)
        a["question"]["ent"] = [token.tag_ for token in doc]
        fo.write('\n' + json.dumps(a) + '\n')
    return fout, tables[table_id]['header']
        
