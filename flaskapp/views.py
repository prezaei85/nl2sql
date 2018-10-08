from flask import render_template
from flask import request
from flaskapp import app
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# import pandas as pd
# import psycopg2


@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')


@app.route('/go')
def go():
    query = request.args.get('query', '')
    


    return render_template(
        'go.html',
        query=query,
    )
