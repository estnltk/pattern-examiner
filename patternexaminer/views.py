import os
import re
import random
import pickle
import glob
import subprocess
from collections import defaultdict
import json

import h5py
import numpy as np
from scipy.special import betaincinv

from sqlalchemy import func, distinct

from flask import render_template, request, jsonify
from patternexaminer import app
from patternexaminer.database import db_session
from patternexaminer.models import InputType, Algorithm, Processing, RegexName,\
        RegexPattern, Experiment, Traceback, WorkExtractedNumber,\
        RawExtractedNumber, PublicData
from patternexaminer.utils import get_model_attr_by_id, get_result_object, get_input_models


LATEST_EXPERIMENTS_COUNT = 50
ALL_CLUSTERS_LABEL = -2


@app.route('/', methods=['GET'])
def index():

    return render_template(
                'clustering.html', 
                sentences_sample=[],
                form_data={
                    'input_types': get_input_types(),
                    'algorithms': get_algorithms(),
                    'regex_names': get_regex_names(),
                    'regex_patterns': get_regex_patterns(),
                    'processing_methods': get_processing_methods()},
                selected_inputs={}
            )


@app.route('/cluster', methods=['POST'])
def cluster():

    process_parameters = [
            'nohup', 
            'python3', 
            'clustering.py',
            '--input_type', 
            request.form['input_type'],
            '--algorithm', 
            request.form['clustering'],
            '--processing_method',
            request.form['preprocessing'],
            '--parent_id',
            request.form['parent_id'],
            '--parent_label',
            request.form['parent_label']
    ]

    if 'regex_name' in request.form:
        process_parameters.extend([
            '--regex_name',
            request.form['regex_name'], 
            '--regex_pattern', 
            request.form['regex_pattern'],
        ])

    subprocess.Popen(process_parameters)

    return jsonify(success=1)


@app.route('/set_evaluation', methods=['POST'])
def set_evaluation():

    result_id = request.form['result_id']
    evaluation = request.form['evaluation']
    experiment_id = request.form['experiment_id']

    Result = get_result_object(experiment_id)
    
    if not evaluation:
        evaluation = None
    result_row = db_session.query(Result).filter(Result.id==result_id).one()
    result_row.evaluation = evaluation
    
    sentence_id = result_row.sentence_id
    # Bubble up the evaluation to all parent experiments.
    parent_experiment_id = db_session.query(Experiment).filter(Experiment.id==result_row.experiment_id).one().parent_id
    while parent_experiment_id:
        parent_row = db_session.query(Result)\
                .filter(Result.experiment_id==parent_experiment_id,
                        Result.sentence_id==sentence_id)\
                .one()
        parent_row.evaluation = evaluation
        parent_experiment_id = db_session.query(Experiment).filter(Experiment.id==parent_row.experiment_id).one().parent_id

    # Float down the evaluation to all child experiments.
    child_experiment_id = db_session.query(Experiment).filter(Experiment.id==result_row.experiment_id).one().child_id
    while child_experiment_id:
        child_row = db_session.query(Result)\
                .filter(Result.experiment_id==child_experiment_id,
                        Result.sentence_id==sentence_id)\
                .one_or_none()
        if not child_row:
            break
        child_row.evaluation = evaluation
        child_experiment_id = db_session.query(Experiment).filter(Experiment.id==child_row.experiment_id).one().child_id

    db_session.commit()

    return jsonify(success=1)


@app.route('/latest_experiments')
def latest_experiments():

    latest_experiments = get_latest_experiments()
    latest_hash = hash(tuple([item for experiment in latest_experiments for item in experiment.values()]))
    return jsonify(
                html=render_template('latest_experiments.html', latest_experiments=latest_experiments),
                latestHash=latest_hash)


@app.route('/clusters_sizes')
def clusters_sizes():

    experiment_id = request.args.get('experiment_id', 0, type=int)

    Result = get_result_object(experiment_id)

    records = db_session.query(Result).filter(Result.experiment_id==experiment_id)
    label_counts = defaultdict(int)
    for rec in records:
        label_counts[rec.label] += 1

    return render_template(
            'cluster_buttons.html', 
            label_counts=label_counts,
            experiment_id=experiment_id,
            all_clusters_label=ALL_CLUSTERS_LABEL)


@app.route('/get_sample')
def get_sample():

    sample_type = request.args.get('type', type=str)
    experiment_id = request.args.get('experiment_id', 0, type=int)
    label = request.args.get('label', 0, type=int)
    sample_size = request.args.get('sample_size', 0, type=int)
    sample_all = request.args.get('sample_all', 0, type=int)
    sample_observed = request.args.get('sample-observed', 0, type=int)

    filtering_type = request.args.get('filtering-type', type=str)
    left_filter = request.args.get('left_filter', type=str)
    content_filter = request.args.get('content_filter', type=str)
    right_filter = request.args.get('right_filter', type=str)

    InputDocument, Result = get_input_models(experiment_id)

    query = db_session.query(Result, InputDocument).join(InputDocument)

    if label == ALL_CLUSTERS_LABEL:
        query = query.filter(Result.experiment_id==experiment_id)
    else:
        query = query.filter(Result.experiment_id==experiment_id, Result.label==label)

    if sample_observed:
        query = query.filter((Result.evaluation==True) | (Result.evaluation==False))
    else:
        query = query.filter(Result.evaluation==None)

    records = query.all()
    if left_filter or content_filter or right_filter:
        filtered_records = []
        if filtering_type == 'or':
            filter_base = [False, False, False]
        if filtering_type == 'and':
            filter_base = [not left_filter, not content_filter, not right_filter]
        for rec in records:
            # match = False
            filter_matches = list(filter_base)
            if left_filter and re.search(left_filter, getattr(rec,InputDocument.__name__).left_context.lower()):
                filter_matches[0] = True
            if content_filter and re.search(content_filter, getattr(rec,InputDocument.__name__).content.lower()):
                filter_matches[1] = True
            if right_filter and re.search(right_filter, getattr(rec,InputDocument.__name__).right_context.lower()):
                filter_matches[2] = True
            
            if filtering_type == 'or':
                match = filter_matches[0] or filter_matches[1] or filter_matches[2]
            if filtering_type == 'and':
                match = filter_matches[0] and filter_matches[1] and filter_matches[2]
            if match:
                filtered_records.append(rec)
        records = filtered_records
        filtered_size = len(records)
    else:
        filtered_size = 0
    
    if sample_type == 'random':
        random.shuffle(records)
        if not sample_all:
            records = records[:sample_size]

    if sample_type == 'heterogenous':
        cached_arrays_id = db_session.query(Experiment).filter(Experiment.id == experiment_id).one().cached_arrays_id
        with h5py.File("cached_arrays.hdf5") as f:
            cached_arrays = f[cached_arrays_id]
            cosine_similarities = np.array(cached_arrays['cosine_similarities'])
            sentence_indexes = np.array(cached_arrays['sentence_indexes'])

        cluster_indexes = []
        result_ids = []
        evaluations = []
        for rec in records:
            cluster_indexes.append(np.where(sentence_indexes==getattr(rec,Result.__name__).sentence_id)[0][0])

        cosine_similarities = cosine_similarities[cluster_indexes][:,cluster_indexes]
        sentence_indexes = np.array(sentence_indexes)[cluster_indexes]

        indexes = np.array([np.random.randint(len(cosine_similarities))])
        for i in range(min(len(cluster_indexes)-1, sample_size-1)):
            distance_array = np.sum(cosine_similarities[indexes], axis=0)
            distance_array[indexes] = np.inf
            argmax = np.argwhere(distance_array == np.min(distance_array)).flatten()
            max_index = np.random.choice(argmax,1)
            indexes = np.append(indexes, max_index)

        # print(indexes)
        homogenous_records = []
        for index in indexes:
            homogenous_records.append(records[index])
        records = homogenous_records

    sentences = []
    for rec in records:
        left_context = clean_text(getattr(rec,InputDocument.__name__).left_context)
        text = clean_text(getattr(rec,InputDocument.__name__).content)
        right_context = clean_text(getattr(rec,InputDocument.__name__).right_context)

        sentences.append({
            'result_id': getattr(rec,Result.__name__).id,
            'event_id': getattr(rec,Result.__name__).sentence_id, 
            'evaluation': getattr(rec,Result.__name__).evaluation, 
            'left_context': left_context,
            'text': text,
            'right_context': right_context
        })

    sentences = sorted(sentences, key=lambda x: x['text'])

    return jsonify(
                html=render_template('get_sample.html', sentences=sentences),
                filteredSize=filtered_size)


def construct_stats(query, experiment_id):
    total = query.count()

    Result = get_result_object(experiment_id)

    positive = query.filter(Result.evaluation==True).count()
    negative = query.filter(Result.evaluation==False).count()
    observed = positive + negative
    positive_bernoulli = bernoulli_trial_probability(positive, observed)
    negative_bernoulli = bernoulli_trial_probability(negative, observed)
    result = {}
    result['positive'] = "{} / {} ({:.2f} - {:.2f})".format(positive, observed, *positive_bernoulli)
    result['negative'] = "{} / {} ({:.2f} - {:.2f})".format(negative, observed, *negative_bernoulli)
    result['observed'] = "{} / {} ({:.0f} %)".format(observed, total, observed/total*100)
    return result


@app.route('/get_statistics')
def get_statistics():

    experiment_id = request.args.get('experiment_id', 0, type=int)
    label = request.args.get('label', ALL_CLUSTERS_LABEL, type=int)

    Result = get_result_object(experiment_id)

    # Experiment stats
    query = db_session.query(Result).filter(Result.experiment_id==experiment_id)
    experiment_stats = construct_stats(query, experiment_id)

    # Cluster stats
    if label != ALL_CLUSTERS_LABEL:
        query = db_session.query(Result).filter(Result.experiment_id==experiment_id, Result.label==label)
        cluster_stats = construct_stats(query, experiment_id)
    else:
        cluster_stats = None

    # Total stats
    parent_experiment_id = db_session.query(Experiment).filter(Experiment.id==experiment_id).one().parent_id
    root_experiment_id = None
    while parent_experiment_id:
        root_experiment_id = parent_experiment_id
        parent_experiment_id = db_session.query(Experiment).filter(Experiment.id==parent_experiment_id).one().parent_id

    if root_experiment_id:
        query = db_session.query(Result).filter(Result.experiment_id==root_experiment_id)
        total_stats = construct_stats(query, experiment_id)
    else:
        total_stats = None

    return render_template('get_statistics.html', total_stats=total_stats, experiment_stats=experiment_stats, cluster_stats=cluster_stats)


@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()


def clean_text(text):
    text = text.replace('\t', '')
    return os.linesep.join([s for s in text.splitlines() if s and s != '\xa0'])


def bernoulli_trial_probability(m, n):
    c = 0.95
    x1 = betaincinv(m + 1, n - m + 1, (1-c)/2)
    x2 = betaincinv(m + 1, n - m + 1, (1+c)/2)
    return x1, x2


def get_latest_experiments():

    records = db_session.query(Experiment, InputType, Algorithm, RegexName, RegexPattern, Processing, Traceback)\
            .join(InputType)\
            .join(Algorithm)\
            .join(RegexPattern, isouter=True)\
            .join(RegexName, isouter=True)\
            .join(Processing)\
            .join(Traceback, isouter=True)\
            .order_by(Experiment.start_time.desc())\
            .limit(LATEST_EXPERIMENTS_COUNT)

    result = []
    for rec in records:

        if rec.Traceback: 
            traceback_message = rec.Traceback.message
        else:
            traceback_message = ''

        if rec.RegexName: 
            regex_name = rec.RegexName.name
            regex_pattern = rec.RegexPattern.pattern
        else:
            regex_name = ''
            regex_pattern = ''

        attributes = {
                'id': rec.Experiment.id,
                'input_type_id': rec.Experiment.input_type,
                'input_type': rec.InputType.name,
                'algorithm_id': rec.Experiment.algorithm,
                'algorithm': rec.Algorithm.name,
                'regex_name_id': rec.Experiment.regex_name,
                'regex_name': regex_name,
                'regex_pattern_id': rec.Experiment.regex_pattern,
                'regex_pattern': regex_pattern,
                'parent_experiment_id': rec.Experiment.parent_id,
                'parent_experiment_label': rec.Experiment.parent_label,
                'processing_id': rec.Experiment.processing,
                'processing': rec.Processing.name,
                'status': rec.Experiment.status,
                'start_time': rec.Experiment.start_time,
                'lines': rec.Experiment.lines,
                'clusters_count': rec.Experiment.clusters_count,
                'traceback': traceback_message
        }
        result.append(attributes)

    return result


def rows_to_list(records, *args):
    result = []
    for rec in records:
        attributes = {'id': rec.id}
        for arg in args:
            attributes[arg] = getattr(rec, arg)   
        result.append(attributes)
    return result


def get_input_types():
    return rows_to_list(InputType.query.all(), 'name')


def get_algorithms():
    return rows_to_list(Algorithm.query.all(), 'name')


def get_regex_names():
    return rows_to_list(RegexName.query.all(), 'name')


def get_regex_patterns():
    return rows_to_list(RegexPattern.query.all(), 'pattern', 'name_id')


def get_processing_methods():
    return rows_to_list(Processing.query.all(), 'name')


