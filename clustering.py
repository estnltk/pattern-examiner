# -*- coding: utf-8 -*-

import os
import re
import time
from optparse import OptionParser
import logging

from collections import defaultdict

import h5py

from joblib import Parallel, delayed

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from hdbscan import HDBSCAN 

from sqlalchemy.sql import func

from patternexaminer.database import db_session
from patternexaminer.models import Experiment, ResultWork, ResultRaw, ResultPublic,\
        Algorithm, Traceback, InputType, RegexName, RegexPattern, WorkExtractedNumber,\
        RawExtractedNumber, PublicData
from patternexaminer.utils import get_model_attr_by_id, get_result_object, get_root_experiment_input_type

# Tables unralated to us give warning, disable it to spare it logging.
import warnings
warnings.filterwarnings("ignore", message="Did not recognize type")

FORMAT = '%(asctime)-15s %(process)d %(levelname)s %(message)s'
logging.basicConfig(filename='cluster_logging.log', format=FORMAT, level=0)
logger = logging.getLogger(__name__)



class DecisionTreeClustering:

    def __init__(self, min_cluster_fraction=0.2, max_depth=5):
        self.min_cluster_fraction = min_cluster_fraction
        self.max_depth = max_depth


    def set_labels(self, rows, path):
        label = int(path)
        for index in rows:
            self.labels_[index] = label 


    def split_goodness(self, index0, index1):
        if len(index0) == 0 or len(index1) == 0:
            return 0
        indx0 = index0.reshape(-1,1)*self.cosine_similarities.shape[1] + index0
        indx1 = index1.reshape(-1,1)*self.cosine_similarities.shape[1] + index1
        # split0 = self.cosine_similarities.take(indx0.flat)
        # split1 = self.cosine_similarities.take(indx1.flat)

        # goodness = np.mean([np.mean(split0), np.mean(split1)])
        goodness = min(len(indx0), len(indx1))
        return goodness


    def decision_walk(self, rows, path='1'):
        
        goodnesses = Parallel(n_jobs=10)(delayed(self.split_goodness)(np.where(self.features[rows,i] == 0)[0], np.where(self.features[rows,i] == 1)[0]) for i in range(self.features.shape[1]))

        goodnesses = np.array(goodnesses)
        argmax = np.argwhere(goodnesses == np.max(goodnesses)).flatten()
        max_index = np.random.choice(argmax,1)
        rows0 = rows[np.argwhere(self.features[rows,max_index] == 0)[:,0]]
        rows1 = rows[np.argwhere(self.features[rows,max_index] == 1)[:,0]]

        if len(path) == self.max_depth:
            self.set_labels(rows1, path=path)
            self.set_labels(rows0, path=path)
            return
        
        if len(rows0) > self.min_cluster_size:
            self.decision_walk(rows0, path=path+'0')
        else:
            self.set_labels(rows0, path=path+'0')

        if len(rows1) > self.min_cluster_size:
            self.decision_walk(rows1, path=path+'1')
        else:
            self.set_labels(rows1, path=path+'1')


    def fit(self, cos_sim, features):
        self.labels_ = np.zeros(len(cos_sim))
        self.features = features
        self.cosine_similarities = cos_sim
        np.fill_diagonal(self.cosine_similarities, 0)

        rows = np.arange(self.features.shape[0])
        self.min_cluster_size = int(len(rows)*self.min_cluster_fraction)
        self.decision_walk(rows)


class Clustering:

    def __init__(self, input_type, algorithm, processing_method, parent_id, parent_label, regex_name=None, regex_pattern=None):
        self.input_type_id = input_type
        self.algorithm_id = algorithm
        self.regex_name_id = regex_name
        self.regex_pattern_id = regex_pattern
        self.processing_method_id = processing_method

        self.parent_id = parent_id
        self.parent_label = parent_label
        # self.algorithm_name = db_session.query(Algorithm).filter(Algorithm.id == self.algorithm_id).one().name
        self.algorithm_name = get_model_attr_by_id(Algorithm, 'name', self.algorithm_id)
        # self.input_type_name = db_session.query(InputType).filter(InputType.id == self.input_type_id).one().name
        self.input_type_name = get_model_attr_by_id(InputType, 'name', self.input_type_id)


    def get_model(self):
        name = self.algorithm_name
        if name == 'AgglomerativeClustering':
            model = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=self.CLUSTERS)
        if name == 'SpectralClustering':
            model = SpectralClustering(affinity='precomputed', n_clusters=self.CLUSTERS)
        if name == 'DBSCAN':
            model = DBSCAN(metric='precomputed')
        if name == 'HDBSCAN':
            model = HDBSCAN(metric='precomputed')
        if name == 'DecisionTree':
            model = DecisionTreeClustering()
        return model


    def sentences_work_extracted_number(self):
        print('sentences_work_extracted_number')
        # regex_name = db_session.query(RegexName).filter(RegexName.id == self.regex_name_id).one().name
        regex_name = get_model_attr_by_id(RegexName, 'name', self.regex_name_id)
        # regex_pattern = db_session.query(RegexPattern).filter(RegexPattern.id == self.regex_pattern_id).one().pattern
        regex_pattern = get_model_attr_by_id(RegexPattern, 'pattern', self.regex_pattern_id)

        if regex_pattern == '*':
            rows = db_session.query(WorkExtractedNumber).filter(WorkExtractedNumber.regex_name == regex_name).all()
        else:
            rows = db_session.query(WorkExtractedNumber)\
                    .filter(WorkExtractedNumber.regex_name == regex_name,
                            WorkExtractedNumber.regex_pattern == regex_pattern)\
                    .all()
        # remove duplicataes
        sentences_no_dups = []
        sentence_indexes = []
        for row in rows:
            sentence = '%s %s' % (row.left_context, row.right_context)
            sentence = sentence.replace('\t', '').replace('\n', '').strip().lower()
            if sentence not in sentences_no_dups:
                sentences_no_dups.append(sentence)
                sentence_indexes.append(row.id)

        return sentences_no_dups, sentence_indexes


    def sentences_extracted_number(self, model_class):
        print('sentences_raw_extracted_number')
        result = db_session.query(model_class).limit(self.experiment.lines)
        sentences = []
        sentence_indexes = []
        for row in result:
            sentences.append("{} {}".format(row.left_context, row.right_context))
            sentence_indexes.append(row.id)
        return sentences, sentence_indexes


    def get_indexed_sentences(self):
        input_type_name = get_root_experiment_input_type(self.experiment.id)
        if input_type_name == 'Extractor':
            return self.sentences_work_extracted_number()
        if input_type_name == 'Raw Data':
            return self.sentences_extracted_number(RawExtractedNumber)
        if input_type_name == 'Sports Data':
            return self.sentences_extracted_number(PublicData)


    def construct_decision_tree_features(self, sentences):
        print('constructing dec_tree features')
        common_patterns = ['([0-9][0-9]\.[0-9][0-9]\.[0-9]{4})', '[0-9][0-9]:[0-9][0-9]:?([0-9][0-9]:?)?', '([0-9][0-9]\.[0-9][0-9]\.[0-9]{4})\s[0-9][0-9]:[0-9][0-9]:?([0-9][0-9]:?)?', '\s*[-/A-ZÜÕÖÄa-züõöä%#,]+', 'Kood ja nimetus:', '[Aa]nalüüsides|seerumis|uriinis|plasmas|S,P-', '([Dd]gn:|[Dd]iagnoos:)', '\([0-9.,]+\s*\.\.\s*[0-9.,]+', '[0-9.,]+\s*[Xx]\s*[0-9.,]+', '[0-9.,]+\s*\([<>][0-9.,]+.*\)', '\s*[-/A-ZÜÕÖÄa-züõöä%#,]+' + '\s*\(.*\):\s*[0-9.,]+', '[mM][cC]?[gG]\s*(\*|x|X|kord[()a]*)\s*[0-9]+', '[0-9]\s*(\*|x|X|kord[()a]*)\s*päevas', '[A-ZÜÕÖÄ0-9 ]+:\s*[-+0-9.,]+', '(ANALÜÜSIDE\sTELLIMUS\snr:|ARHIIVI NR)', '([0-9][0-9]\.[0-9][0-9]\.[0-9]{4})\s[0-9][0-9]:[0-9][0-9]:?([0-9][0-9]:?)?\s\s*[-/A-ZÜÕÖÄa-züõöä%#,]+', '[0-9.,]+\s*\([^.]+\)', '(^|\s)[A-Z][a-z]+:\s[0-9]', '[0-9]+\s+[0-9]+\s+[0-9]+', '[0-9,.]+\s*[A-ZÜÕÖÄa-züõöäµ]+/[A-ZÜÕÖÄa-züõöäµ]+']
        if self.input_type_name == 'Sports Data':
            common_patterns = ['Eesti', 'aasta', 'punkt', 'minut', 'meetri', 'Poola', 'Tartu', 'Soome', 'koha', 'võitis']
        regexes_features = np.zeros((len(sentences), len(common_patterns)))
        for sentence_index, sentence in enumerate(sentences):
            for feature_index, pattern in enumerate(common_patterns):
                # if re.search(pattern, sentence):
                if re.search(pattern, sentence.lower()):
                    regexes_features[sentence_index][feature_index] = 1

        units_filename = 'units.tsv'
        if os.path.isfile(units_filename):
            clean_units = set()
            mapped_units = {}
            with open(units_filename) as f:
                for line in f:
                    units = line.rstrip().split('\t')
                    mapped_units[units[0]] = units[1]
                    clean_units.add(units[1])

            clean_index = {}
            for i, unit in enumerate(clean_units):
                clean_index[unit] = i

            units_features = np.zeros((len(sentences), len(clean_index)))
            for sentence_index, sentence in enumerate(sentences):
                for dirty_unit, clean_unit in mapped_units.items():
                    if dirty_unit in sentence:
                        units_features[sentence_index][clean_index[clean_unit]] = 1

            features = np.hstack([regexes_features, units_features])
        else:
            features = regexes_features

        return features


    def construct_cos_sim(self, sentences):
        print('constructing cos_sim')
        tfidf_vectorizer = TfidfVectorizer()
        tfidf = tfidf_vectorizer.fit_transform(sentences)
        count = tfidf.shape[0]
        cosine_similarities = np.zeros((count, count))
        for i in range(count):
            cosine_similarities[i,:] = linear_kernel(tfidf[i], tfidf).flatten()
        return cosine_similarities


    def validate_experiment(self, sentence_indexes):
        print('validating experimnet')
        lines = len(sentence_indexes)
        exception = Exception('Too many lines for algorithm')
        if lines > 70000:
            raise exception
        if lines > 25000 and self.algorithm_name == 'AgglomerativeClustering':
            raise exception
        if lines > 15000 and self.algorithm_name == 'SpectralClustering':
            raise exception


    def save_labels(self, sentence_indexes, labels):
        print('saving labels')
        Result = get_result_object(self.experiment.id)
        # Get the parent experiment evaluations
        evaluations = defaultdict(lambda: None)
        if self.experiment.parent_id:
            parent_results = db_session.query(Result).filter(Result.experiment_id==self.experiment.parent_id).all()
            for parent_result in parent_results:
                evaluations[parent_result.sentence_id] = parent_result.evaluation
        
        for sentence_index, label in zip(sentence_indexes, labels):
            sentence_id = int(sentence_index)
            db_session.add(Result(experiment_id=self.experiment.id, label=int(label), 
                    sentence_id=sentence_id, evaluation=evaluations[sentence_id]))
        self.experiment.clusters_count = len(set(labels))
        db_session.commit()


    def cache_features(self, cache_id, sentences):
        print('cache features to the array db object')
        features = self.construct_decision_tree_features(sentences)
        CACHED_ARRAYS[str(cache_id)]['features'] = features


    def construct_cached_arrays(self, sentences, sentence_indexes):
        print('construct the cached arrays')

        # Maybe move this somewhere else?
        self.validate_experiment(sentence_indexes)

        cosine_similarities = self.construct_cos_sim(sentences)

        sentence_indexes = np.array(sentence_indexes)

        cache_id = str(self.experiment.id)
        CACHED_ARRAYS.create_group(cache_id)
        CACHED_ARRAYS[cache_id]['cosine_similarities'] = cosine_similarities
        CACHED_ARRAYS[cache_id]['sentence_indexes'] = sentence_indexes

        if self.algorithm_name == 'DecisionTree':
            self.cache_features(cache_id, sentences)

        self.experiment.cached_arrays_id = cache_id

        return CACHED_ARRAYS[cache_id]


    def h5_data_to_numpy(self, cached_arrays):
        arrays = defaultdict(lambda: None)
        arrays['cosine_similarities'] = np.array(cached_arrays['cosine_similarities'])
        arrays['sentence_indexes'] = np.array(cached_arrays['sentence_indexes'])
        
        if self.algorithm_name == 'DecisionTree':
            if 'features' not in cached_arrays:
                sentences, _ = self.get_indexed_sentences()
                self.cache_features(self.experiment.cached_arrays_id, sentences)
            arrays['features'] = np.array(cached_arrays['features'])
        
        return arrays


    def similar_experiment_arrays(self, filter_args):

        similar_experiment_query = db_session.query(Experiment)\
                .filter(Experiment.id!=self.experiment.id,
                        Experiment.input_type==self.input_type_id,
                        Experiment.processing==self.processing_method_id,
                        Experiment.cached_arrays_id!=None)

        for args in filter_args:
            similar_experiment_query = similar_experiment_query.filter(args[0]==args[1])

        similar_experiment = similar_experiment_query.first()

        if similar_experiment:
            print('found similar experiment, getting arrays from db')
            cache_id = similar_experiment.cached_arrays_id
            cached_arrays = CACHED_ARRAYS[str(cache_id)]
            self.experiment.cached_arrays_id = cache_id

        else:
            print('no such experiment, making new arrays')
            sentences, sentence_indexes = self.get_indexed_sentences()
            cached_arrays = self.construct_cached_arrays(sentences, sentence_indexes)

        arrays = self.h5_data_to_numpy(cached_arrays)

        return arrays


    def run(self):
        try:
            # Log the self.experiment
            start_time = time.time()
            self.experiment = Experiment(
                            input_type=self.input_type_id,
                            algorithm=self.algorithm_id,
                            processing=self.processing_method_id,
                            start_time=func.current_timestamp(),
                            status='running')

            db_session.add(self.experiment)
            db_session.commit()

            # START PREPROCESSING

            if self.input_type_name == 'Extractor':

                self.experiment.regex_name = self.regex_name_id
                self.experiment.regex_pattern = self.regex_pattern_id
                db_session.commit()

                arrays = self.similar_experiment_arrays(
                        filter_args=[(Experiment.regex_name, self.regex_name_id),
                                     (Experiment.regex_pattern, self.regex_pattern_id)])
                cosine_similarities = arrays['cosine_similarities']
                sentence_indexes = arrays['sentence_indexes']
                features = arrays['features']

                self.experiment.lines = len(sentence_indexes)


            if self.input_type_name == 'Cluster':

                parent = db_session.query(Experiment).filter(Experiment.id == self.parent_id).one()
                parent.child_id = self.experiment.id

                self.experiment.parent_id = self.parent_id
                self.experiment.parent_label = self.parent_label
                self.experiment.regex_name = parent.regex_name
                self.experiment.regex_pattern = parent.regex_pattern
                
                cached_arrays_id = parent.cached_arrays_id
                self.experiment.cached_arrays_id = cached_arrays_id

                # cached_arrays = db_session.query(CachedArrays).filter(CachedArrays.id==cached_arrays_id).one()
                cached_arrays = CACHED_ARRAYS[cached_arrays_id]
                
                arrays = self.h5_data_to_numpy(cached_arrays)
                cosine_similarities = arrays['cosine_similarities']
                sentence_indexes = arrays['sentence_indexes']
                features = arrays['features']

                Result = get_result_object(parent.id)
                rows = db_session.query(Result).filter(Result.experiment_id == self.parent_id, Result.label == self.parent_label)
                cluster_indexes = []

                sentence_indexes_list = sentence_indexes.tolist()
                for row in rows:
                    cluster_indexes.append(sentence_indexes_list.index(row.sentence_id))

                cosine_similarities = cosine_similarities[cluster_indexes][:,cluster_indexes]
                sentence_indexes = sentence_indexes[cluster_indexes]
                if self.algorithm_name == 'DecisionTree':
                    features = features[cluster_indexes,:]
                
                self.experiment.lines = len(sentence_indexes)

            if self.input_type_name == 'Raw Data' or self.input_type_name == 'Sports Data':
                print(self.input_type_name)
                self.experiment.lines = 602
                db_session.commit()

                arrays = self.similar_experiment_arrays(filter_args=[(Experiment.lines, self.experiment.lines)])

                cosine_similarities = arrays['cosine_similarities']
                sentence_indexes = arrays['sentence_indexes']
                features = arrays['features']


            db_session.commit()

            # raise Exception('debugging')

            # END PREPROCESSING

            # Log preprocessing duration.
            self.experiment.preprocessing_seconds = time.time() - start_time
            db_session.commit()


            # START CLUSTERING
            print('starting clustering')
            clustering_start = time.time()

            self.CLUSTERS = min(20, len(sentence_indexes))
            model = self.get_model()

            if self.algorithm_name != 'DecisionTree':
                model.fit(cosine_similarities)
            else:
                model.fit(cosine_similarities, features)

            self.save_labels(sentence_indexes, model.labels_)
            # END CLUSTERING

            # Log clustering duration.
            self.experiment.clustering_seconds = time.time() - clustering_start
            self.experiment.status = 'finished'
            db_session.commit()
            logger.debug('finished')

        except Exception as e:
            self.experiment.status = 'error'
            db_session.add(Traceback(experiment_id=self.experiment.id, message=str(e)))
            logger.exception(str(e))
            db_session.commit()
            CACHED_ARRAYS.close()
            raise e


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--input_type', type="int")
    parser.add_option('--algorithm', type="int")
    parser.add_option('--regex_name', type="int")
    parser.add_option('--regex_pattern', type="int")
    parser.add_option('--processing_method', type="int")
    parser.add_option('--parent_id', type="int")
    parser.add_option('--parent_label', type="int")

    (options, args) = parser.parse_args()
    input_type_id = options.input_type
    algorithm_id = options.algorithm
    regex_name_id = options.regex_name
    regex_pattern_id = options.regex_pattern
    processing_method_id = options.processing_method

    parent_id = options.parent_id
    parent_label = options.parent_label

    clustering = Clustering(
        input_type=input_type_id,
        algorithm=algorithm_id,
        processing_method=processing_method_id,
        regex_name=regex_name_id,
        regex_pattern=regex_pattern_id,
        parent_id=parent_id,
        parent_label=parent_label
    )

    CACHED_ARRAYS = h5py.File('cached_arrays.hdf5')

    # if 'regex_name' in request.form:
    clustering.run()
    CACHED_ARRAYS.close()