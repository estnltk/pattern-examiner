from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, ForeignKeyConstraint, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship
from patternexaminer.database import Base
from patternexaminer.config import config


class PatternExaminerBase(Base):
    __abstract__ = True

    _table_prefix = config.get('DATABASE','TABLE_PREFIX')

    # @declared_attr
    # def id(cls):
    #     return Column(Integer, primary_key=True)

    id = Column(Integer, primary_key=True)

    # All these tables have a common prefix.
    @declared_attr
    def __tablename__(cls):
        return cls._table_prefix + cls.table_name


class InputType(PatternExaminerBase):
    table_name = 'input_type'
    name = Column(String)


class Algorithm(PatternExaminerBase):
    table_name = 'algorithm'
    name = Column(String)


class Processing(PatternExaminerBase):
    table_name = 'processing'
    name = Column(String)


class RegexName(PatternExaminerBase):
    table_name = 'regex_name'
    name = Column(String)


class RegexPattern(PatternExaminerBase):
    table_name = 'regex_pattern'
    pattern = Column(String)
    name_id = Column(Integer, ForeignKey(RegexName.id))


# class CachedArrays(PatternExaminerBase):
#     table_name = 'cached_arrays'
#     cosine_similarities = Column(LargeBinary)
#     cosine_similarities_dtype = Column(String)
#     cosine_similarities_dim = Column(Integer)
#     sentence_indexes = Column(LargeBinary)
#     sentence_indexes_dtype = Column(String)
#     features = Column(LargeBinary)
#     features_dtype = Column(String)
#     features_rows = Column(Integer)
#     features_cols = Column(Integer)


class Experiment(PatternExaminerBase):
    table_name = 'experiment'
    input_type = Column(Integer, ForeignKey(InputType.id))
    algorithm = Column(Integer, ForeignKey(Algorithm.id))
    regex_name = Column(Integer, ForeignKey(RegexName.id))
    regex_pattern = Column(Integer, ForeignKey(RegexPattern.id))
    parent_id = Column(Integer, ForeignKey('{}{}.id'.format(PatternExaminerBase._table_prefix, table_name)))
    parent_label = Column(Integer)
    raw_lines_count = Column(Integer)
    child_id = Column(Integer, ForeignKey('{}{}.id'.format(PatternExaminerBase._table_prefix, table_name)))
    processing = Column(Integer, ForeignKey(Processing.id))
    start_time = Column(DateTime)
    preprocessing_seconds = Column(Integer)
    clustering_seconds = Column(Integer)
    status = Column(String)
    lines = Column(Integer)
    clusters_count = Column(Integer)
    cached_arrays_id = Column(String)

    # Not sure if parent_id foreign key works without this, but id can't be 
    # read from the PatternExaminerBase class at the moment.
    # parent = relationship('Experiment', remote_side=[id])

    
# The common class for all the input types we wish to cluster.
class InputDocument(object):
    left_context = Column(String)
    content = Column(String)
    right_context = Column(String)
    pass


class WorkExtractedNumber(InputDocument, PatternExaminerBase):
    table_name = 'work_extracted_number'
    regex_name = Column(String)
    regex_pattern = Column(String)


class RawExtractedNumber(InputDocument, PatternExaminerBase):
    table_name = 'raw_extracted_number'
    table = Column(String)
    field = Column(String)


class PublicData(InputDocument, PatternExaminerBase):
    table_name = 'public_data'


class Result(object):
    label = Column(Integer)
    @declared_attr
    def experiment_id(cls):
        return Column(Integer, ForeignKey(Experiment.id))
    evaluation = Column(Boolean)


class ResultWork(Result, PatternExaminerBase):
    table_name = 'result_work'
    sentence_id = Column(Integer, ForeignKey(WorkExtractedNumber.id))


class ResultRaw(Result, PatternExaminerBase):
    table_name = 'result_raw'
    sentence_id = Column(Integer, ForeignKey(RawExtractedNumber.id))


class ResultPublic(Result, PatternExaminerBase):
    table_name = 'result_public'
    sentence_id = Column(Integer, ForeignKey(PublicData.id))


class Traceback(PatternExaminerBase):
    table_name = 'traceback'
    experiment_id = Column(Integer, ForeignKey(Experiment.id))
    message = Column(String)

