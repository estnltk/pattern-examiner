from patternexaminer.database import db_session
from patternexaminer.models import ResultWork, ResultRaw, ResultPublic,\
        WorkExtractedNumber, RawExtractedNumber, PublicData, InputType, Experiment

# IN VIEWS, ADD THIS EVERYWHERE!
def get_model_attr_by_id(model, attribute, index):
    rec = db_session.query(model).filter(model.id == index).one()
    return getattr(rec, attribute)


def get_root_experiment_input_type(experiment_id):
    experiment = db_session.query(Experiment).filter(Experiment.id==experiment_id).one()
    parent_experiment_id = db_session.query(Experiment).filter(Experiment.id==experiment_id).one().parent_id
    if parent_experiment_id:
        while parent_experiment_id:
            experiment = db_session.query(Experiment).filter(Experiment.id==parent_experiment_id).one()
            parent_experiment_id = experiment.parent_id
    input_type_id = experiment.input_type
    input_type_name = get_model_attr_by_id(InputType, 'name', input_type_id)
    return input_type_name


def get_input_models(experiment_id):
    input_type_name = get_root_experiment_input_type(experiment_id)
    if input_type_name == 'Extractor':
        return WorkExtractedNumber, ResultWork
    if input_type_name == 'Raw Data':
        return RawExtractedNumber, ResultRaw
    if input_type_name == 'Sports Data':
        return PublicData, ResultPublic


def get_result_object(experiment_id):
    return get_input_models(experiment_id)[1]

