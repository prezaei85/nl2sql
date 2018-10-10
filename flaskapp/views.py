from flask import render_template, request, flash, redirect,    url_for
from flaskapp import app
from flaskapp.helpers import get_table_data
import json
import sys
import subprocess
sys.path.append("./src")
from annotate_question import annotate_question
from evaluate_question import main

@app.route('/')
@app.route('/index')
def index():
    with open('config/app_config.json', 'r') as f:
        con = json.load(f)

    questions, header, data = get_table_data(
        con["table_file"], con["question_file"], con["table_id"], con["question_indices"])

    return render_template('master.html',
        questions = questions, header = header, data = data
    )

@app.route('/go')
def go():
    question = request.args.get('question', '')

    if not question:
        flash('Please enter a question.')
        return redirect(url_for('index'))

    with open('config/app_config.json', 'r') as f:
        con = json.load(f)

    questions, header, data = get_table_data(
        con["table_file"], con["question_file"], con["table_id"], con["question_indices"])

    anno_filename, headers = annotate_question(
        question, con["table_id"], con["dir_in"], con["dir_out"], con["split"])

    args = ['-model_path', con["model"], '-data_path', con["data_path"], \
        "-anno_data_path", con["anno_path"]]
    sql_query, result_list = main(
        anno_filename, headers, args)

    if result_list is None:
        result = 'An error occured when executing the SQL query.'
    else:
        result = ', '.join([str(x) for x in result_list])

    return render_template('go.html', 
        header = header, data = data, question = question, sql_query = sql_query, result = result
    )

