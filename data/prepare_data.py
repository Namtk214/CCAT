# -*- coding: utf-8 -*-
import os
import csv
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import ast
from setting import *


def stat_unique(data: pd.DataFrame, key):
    if key is None:
        print('Total length: {}'.format(len(data)))
    elif isinstance(key, str):
        print('Number of unique {}: {}'.format(key, len(data[key].unique())))
    elif isinstance(key, list):
        print('Number of unique [{}]: {}'.format(
            ','.join(key), len(data.drop_duplicates(key, keep='first'))))


def parse_data(data):
    """ 
    Args:
        data: list of triplets (sid, qid, score)
    Returns:
        student based datasets: defaultdict {sid: {qid: score}}
        question based datasets: defaultdict {qid: {sid: score}}
    """
    stu_data = defaultdict(lambda: defaultdict(dict))
    ques_data = defaultdict(lambda: defaultdict(dict))
    for i, row in data.iterrows():
        sid = row.student_id
        qid = row.question_id
        correct = row.correct
        stu_data[sid][qid] = correct
        ques_data[qid][sid] = correct
    return stu_data, ques_data


def renumber_student_id(data):
    """
    Args:
        data: list of triplets (sid, qid, score)
    Returns:
        renumbered datasets: list of triplets (sid, qid, score)
    """
    student_ids = sorted(set(t[0] for t in data))
    renumber_map = {sid: i for i, sid in enumerate(student_ids)}
    data = [(renumber_map[t[0]], t[1], t[2]) for t in data]
    return data


def save_to_csv(data, path):
    """
    Args:
        data: list of triplets (sid, qid, correct)
        path: str representing saving path
    """
    pd.DataFrame.from_records(sorted(data), columns=[
                              'student_id', 'question_id', 'correct']).to_csv(path, index=False)


#############################################
# Preprocess dành cho NIPS2020
#############################################
if params.data_name == 'NIPS2020':
    # Đọc dữ liệu gốc và đổi tên cột
    raw_data = pd.read_csv('NIPS2020/train_task_3_4.csv',
                           encoding='ISO-8859-1')
    raw_data.head()
    raw_data = raw_data.rename(columns={'UserId': 'student_id',
                                        'QuestionId': 'question_id',
                                        'IsCorrect': 'correct',
                                        })
    # Lấy các cột cần thiết: student_id, question_id, correct, CorrectAnswer
    all_data = raw_data.loc[:, ['student_id',
                                'question_id', 'correct', 'CorrectAnswer']].dropna()

    stat_unique(all_data, None)
    stat_unique(all_data, ['student_id', 'question_id'])
    stat_unique(all_data, 'student_id')
    stat_unique(all_data, 'question_id')
    # Lọc dữ liệu theo số lượt trả lời
    n_students = all_data.groupby('question_id')['student_id'].count()
    question_filter = n_students[n_students < 50].index.tolist()
    print(f'filter {len(question_filter)} questions')
    selected_data = all_data[~all_data['question_id'].isin(question_filter)]

    n_questions = selected_data.groupby('student_id')['question_id'].nunique()
    student_filter = n_questions[n_questions < 50].index.tolist()
    print(f'filter {len(student_filter)} students')
    selected_data = selected_data[~selected_data['student_id'].isin(
        student_filter)]

    # Renumber student_id
    s2n = {}
    cnt = 0
    for i, row in selected_data.iterrows():
        if row.student_id not in s2n:
            s2n[row.student_id] = cnt
            cnt += 1
    selected_data.loc[:, 'student_id'] = selected_data.loc[:,
                                                           'student_id'].apply(lambda x: s2n[x])

    # Renumber question_id và xây dựng mapping q2n:
    q2n = {}
    cnt = 0
    for i, row in selected_data.iterrows():
        if row.question_id not in q2n:
            q2n[row.question_id] = cnt
            cnt += 1
    selected_data.loc[:, 'question_id'] = selected_data.loc[:,
                                                            'question_id'].apply(lambda x: q2n[x])

    stat_unique(selected_data, None)
    stat_unique(selected_data, ['student_id', 'question_id'])
    stat_unique(selected_data, 'student_id')
    stat_unique(selected_data, 'question_id')

    # Split dữ liệu thành train, test, all
    data = []
    for i, row in selected_data.iterrows():
        data.append([row.student_id, row.question_id, row.correct])

    stu_data, ques_data = parse_data(selected_data)
    test_size = 0.2
    least_test_length = 150
    random.seed(2024)

    n_students = len(stu_data)
    if isinstance(test_size, float):
        test_size = int(n_students * test_size)
    train_size = n_students - test_size
    assert (train_size > 0 and test_size > 0)

    students = list(range(n_students))
    random.shuffle(students)
    if least_test_length is not None:
        student_lens = defaultdict(int)
        for t in data:
            student_lens[t[0]] += 1
        students = [student for student in students
                    if student_lens[student] >= least_test_length]
    test_students = set(students[:test_size])

    train_data = [record for record in data if record[0] not in test_students]
    test_data = [record for record in data if record[0] in test_students]

    train_data = renumber_student_id(train_data)
    test_data = renumber_student_id(test_data)
    all_data = renumber_student_id(data)

    print(f'train records length: {len(train_data)}')
    print(f'test records length: {len(test_data)}')
    print(f'all records length: {len(all_data)}')
    # Lưu data ra file CSV
    save_to_csv(train_data, 'NIPS2020/train_triples.csv')
    save_to_csv(test_data, 'NIPS2020/test_triples.csv')
    save_to_csv(all_data, 'NIPS2020/triples.csv')
    metadata = {"num_students": 4914,
                "num_questions": 900,
                "num_records": 1382173,
                "num_train_students": 3932,
                "num_test_students": 982}

    with open('NIPS2020/metadata.json', 'w') as f:
        json.dump(metadata, f)

    #############################################
    # Tạo file preprocessed_questions.csv mới
    #############################################

    # Đọc lại file train_task_3_4.csv gốc để lấy thông tin về CorrectAnswer và các cột liên quan
    df_orig = pd.read_csv('NIPS2020/train_task_3_4.csv', encoding='ISO-8859-1')
    # Chuyển tên cột cho phù hợp
    df_orig = df_orig.rename(columns={'UserId': 'student_id',
                                      'QuestionId': 'question_id',
                                      'IsCorrect': 'correct'})

    # Chọn các cột cần thiết: question_id và CorrectAnswer, và loại bỏ dòng trùng theo question_id
    df_q = df_orig.loc[:, ['question_id', 'CorrectAnswer']
                       ].dropna().drop_duplicates(subset=['question_id'])

    # Áp dụng cùng mapping renumbering q2n cho question_id.
    # Chỉ giữ lại những câu hỏi có trong q2n (đã được tạo ở phần xử lý trước).
    df_q = df_q[df_q['question_id'].isin(q2n.keys())].copy()
    df_q['new_question_id'] = df_q['question_id'].apply(lambda x: q2n[x])

    # Giữ đúng đáp án theo dạng số (1,2,3,4) – không chuyển đổi mapping
    df_q['CorrectAnswer'] = df_q['CorrectAnswer'].astype(str).str.strip()

    # Lấy ánh xạ từ file question_metadata_task_3_4.csv để có subject ids theo old question_id
    df_qmeta = pd.read_csv(
        'NIPS2020/question_metadata_task_3_4.csv', encoding='ISO-8859-1')
    df_qmeta = df_qmeta[['QuestionId', 'SubjectId']].drop_duplicates()
    mapping_qmeta = {}
    for i, row in df_qmeta.iterrows():
        try:
            orig_qid = int(row['QuestionId'])
            # Parse chuỗi dạng "[3, 71, 98, 209]" về list các số
            subject_ids = ast.literal_eval(row['SubjectId'])
            mapping_qmeta[orig_qid] = subject_ids
        except Exception as e:
            continue

    # Tạo dictionary ánh xạ từ new question id sang subject ids, dựa trên mapping q2n
    new_subject_mapping = {}
    for orig_qid, subject_ids in mapping_qmeta.items():
        if orig_qid in q2n:
            new_qid = q2n[orig_qid]
            new_subject_mapping[new_qid] = subject_ids

    # Đọc file subject_metadata.csv để có ánh xạ từ SubjectId sang subject name
    df_smeta = pd.read_csv('NIPS2020/subject_metadata.csv', encoding='utf-8')
    col_name = 'subjectid'
    if col_name not in df_smeta.columns:
        col_name = 'SubjectId'
    subject_meta_dict = {}
    for i, row in df_smeta.iterrows():
        sid = int(row[col_name])
        subject_meta_dict[sid] = row['Name']

    # Hàm chuyển đổi danh sách subject ids thành chuỗi tên các thể loại (cách nhau bởi dấu phẩy)
    def get_subject_names(subj_ids):
        return ", ".join([subject_meta_dict.get(sid, str(sid)) for sid in subj_ids])

    # Tạo cột SubjectsNames từ new_subject_mapping (dựa trên new_question_id)
    df_q['SubjectsNames'] = df_q['new_question_id'].apply(
        lambda new_qid: get_subject_names(new_subject_mapping.get(new_qid, []))
    )

    # Để lưu lại thông tin câu hỏi gốc (OldQuestionId), tạo mapping inverse của q2n:
    inverse_q2n = {v: k for k, v in q2n.items()}
    df_q['OldQuestionId'] = df_q['new_question_id'].apply(
        lambda x: inverse_q2n.get(x, "")
    )

    # Lựa chọn các cột theo thứ tự: OldQuestionId, new_question_id, SubjectsNames, CorrectAnswer
    df_q_final = df_q[['OldQuestionId', 'new_question_id',
                       'SubjectsNames', 'CorrectAnswer']]

    # Giới hạn chỉ lấy 900 câu theo new_question_id đã sắp xếp (tùy chỉnh theo yêu cầu)
    df_q_final = df_q_final.sort_values(by='new_question_id').head(900)

    # Lưu kết quả ra file preprocessed_questions.csv
    output_path = os.path.join("NIPS2020", "preprocessed_questions.csv")
    df_q_final.to_csv(output_path, index=False)
    print(f"Đã tạo file {output_path} với {len(df_q_final)} câu hỏi.")
    #############################################
    # Tiếp theo, xử lý dữ liệu cho JUNYI (nếu cần)...
    #############################################

if params.data_name == 'JUNYI':
    # Các xử lý dữ liệu cho JUNYI (code đã có)
    raw_data = pd.read_csv(
        'JUNYI/junyi_ProblemLog_original.csv', encoding='ISO-8859-1')
    raw_data.head()
    raw_data = raw_data.rename(columns={'user_id': 'student_id',
                                        'exercise': 'question_id',
                                        })
    all_data = raw_data.loc[:, [
        'student_id', 'question_id', 'correct', 'problem_number']].dropna()
    stat_unique(all_data, None)
    stat_unique(all_data, ['student_id', 'question_id'])
    stat_unique(all_data, 'student_id')
    stat_unique(all_data, 'question_id')
    # filter data
    selected_data = all_data
    n_students = selected_data.groupby('question_id')['student_id'].count()
    question_filter = n_students[n_students < 50].index.tolist()
    print(f'filter {len(question_filter)} questions')
    selected_data = selected_data[~selected_data['question_id'].isin(
        question_filter)]

    n_questions = selected_data.groupby('student_id')['question_id'].nunique()
    student_filter = n_questions[n_questions < 50].index.tolist()
    print(f'filter {len(student_filter)} students')
    selected_data = selected_data[~selected_data['student_id'].isin(
        student_filter)]

    selected_groupby = selected_data[['student_id', 'question_id', 'problem_number']].groupby(
        by=['student_id', 'question_id'], as_index=False).max()
    selected_merge = pd.merge(selected_groupby, selected_data, on=[
                              'student_id', 'question_id', 'problem_number'], how='left')
    q2n = {}
    cnt = 0
    for i, row in selected_merge.iterrows():
        if row.question_id not in q2n:
            q2n[row.question_id] = cnt
            cnt += 1
    selected_merge.loc[:, 'question_id'] = selected_merge.loc[:,
                                                              'question_id'].apply(lambda x: q2n[x])
    s2n = {}
    cnt = 0
    for i, row in selected_merge.iterrows():
        if row.student_id not in s2n:
            s2n[row.student_id] = cnt
            cnt += 1
    selected_merge.loc[:, 'student_id'] = selected_merge.loc[:,
                                                             'student_id'].apply(lambda x: s2n[x])
    knowledge_data = pd.read_csv("JUNYI/junyi_Exercise_table.csv")
    k2n = {}
    cnt = 0
    for i, row in knowledge_data.iterrows():
        if row.topic not in k2n:
            k2n[row.topic] = cnt
            cnt += 1
    knowledge_data.loc[:, 'topic'] = knowledge_data.loc[:,
                                                        'topic'].apply(lambda x: k2n[x])
    q2k = {}
    table = knowledge_data.loc[:, ['name', 'topic']].drop_duplicates()
    for i, row in table.iterrows():
        q = row['name']
        if q in q2n:
            q2k[int(q2n[q])] = row['topic']
    with open('JUNYI/concept_map.json', 'w') as f:
        json.dump(q2k, f)
    stat_unique(selected_data, None)
    stat_unique(selected_data, ['student_id', 'question_id'])
    stat_unique(selected_data, 'student_id')
    stat_unique(selected_data, 'question_id')

    # split data
    data = []
    for i, row in selected_merge.iterrows():
        data.append([row.student_id, row.question_id, row.correct])

    stu_data, ques_data = parse_data(selected_merge)
    test_size = 0.2
    least_test_length = None
    random.seed(2024)

    n_students = len(stu_data)
    if isinstance(test_size, float):
        test_size = int(n_students * test_size)
    train_size = n_students - test_size
    assert (train_size > 0 and test_size > 0)

    students = list(range(n_students))
    random.shuffle(students)
    if least_test_length is not None:
        student_lens = defaultdict(int)
        for t in data:
            student_lens[t[0]] += 1
        students = [student for student in students
                    if student_lens[student] >= least_test_length]
    test_students = set(students[:test_size])

    train_data = [record for record in data if record[0] not in test_students]
    test_data = [record for record in data if record[0] in test_students]

    train_data = renumber_student_id(train_data)
    test_data = renumber_student_id(test_data)
    all_data = renumber_student_id(data)

    print(f'train records length: {len(train_data)}')
    print(f'test records length: {len(test_data)}')
    print(f'all records length: {len(all_data)}')
    # save data
    save_to_csv(train_data, 'JUNYI/train_triples.csv')
    save_to_csv(test_data, 'JUNYI/test_triples.csv')
    save_to_csv(all_data, 'JUNYI/triples.csv')
    metadata = {"num_students": 8852,
                "num_questions": 702,
                "num_records": 801270,
                "num_train_students": 7082,
                "num_test_students": 1770}
    with open('JUNYI/metadata.json', 'w') as f:
        json.dump(metadata, f)
