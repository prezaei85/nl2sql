import sys
import json
sys.path.append("./src")
from annotate_question import annotate_question
from evaluate_question import main

question = "How many times did the Seattle Seahawks played at Kingdome?"
table_id = "2-13258745-2" 
dataset = "test"

with open('config/config.json', 'r') as f:
    con = json.load(f)

anno_filename, headers = annotate_question(
	question, table_id, con["dir_in"], con["dir_out"], dataset)

args = ['-model_path', con["model"], '-data_path', con["data_path"], \
        "-anno_data_path", con["anno_path"]]
sql_code, result_list = main(anno_filename, headers, args)

print("SQL code:", sql_code)
print("Execution result:", ', '.join([str(x) for x in result_list]))
