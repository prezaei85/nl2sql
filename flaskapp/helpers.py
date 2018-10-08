import json

def get_table_data(table_file, question_file, table_id, question_indices = []):
    tables_by_id = {}
    with open(table_file) as ft:
        for lt in ft:
            eg = json.loads(lt)
            tables_by_id[eg['id']] = eg

    questions_by_tabe_id = {}
    with open(question_file) as fs:
        for ls in fs:
            eg = json.loads(ls)
            tid = eg['table_id']
            if tid not in questions_by_tabe_id:
                questions_by_tabe_id[tid] = []
            questions_by_tabe_id[tid].append(eg['question'])

    header = tables_by_id[table_id]['header']
    data = tables_by_id[table_id]['rows']
    if question_indices:
    	questions = [questions_by_tabe_id[table_id][i] for i in question_indices]    	
    else:
    	questions = questions_by_tabe_id[table_id]
    return questions, header, data