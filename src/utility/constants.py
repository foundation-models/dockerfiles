
from enum import Enum


class MlflowStage(Enum):
    production = 'Production'
    staging = 'Staging'
    development = 'Development'
    none = 'None'


class HuggingFaceRepository(Enum):
    """Defines the HuggingFace repository names."""
    # The HuggingFace repository names.
    use_auth_token = 'use_auth_token'


class Loggers(Enum):
    """Enum for the different loggers used in the project."""
    mlflow = "mlflow"
    tensorboard = "tensorboard"


class CheckpointCallbacks(Enum):
    """Enum for the different checkpoint callbacks used in the project."""
    mlflow = "mlflow_checkpoint"
    early_stopping = "early_stopping"
    model_checkpoint = "model_checkpoint"


class TokenClassificationColumn(Enum):
    """Enum for token classification columns."""
    tokens = 'tokens'
    tags = 'ner_tags'
    token_count = 'token_count'


class LabelColumns(Enum):
    """Enum for label columns."""
    single_processing = 'entity'
    batch_processing = 'entity_group'


class ControlCharacters(Enum):
    new_line = '\n'
    tab = '\t'
    space = ' '


class CommonColumn(Enum):
    id = 'id'
    text = 'text'
    language = 'language'


class NERColumn(Enum):
    id = 'ID'
    name = 'name'
    entity = 'entity'
    score = 'score'


class JsonRequestKeys(Enum):
    data = 'data'


class DatasetsSplit(Enum):
    train = 'train'
    test = 'test'
    validation = 'validation'
    predict = 'predict'


class Flag(Enum):
    perfect_match = 'perfect_match'
    imperfect_match = 'imperfect_match'
    no_match = 'no_match'
    too_noisy_match = 'too_noisy_match'


class SignatureColumn(Enum):
    email_id = 'email_id'
    signature = 'signature'
    tag = 'tag'
    signature_candidate = 'signature_candidates'
    signature_token_count = 'signature_token_count'
    start_signature = 'start_signature'
    end_signature = 'end_signature'
    score_signature = 'score_signature'
    row_start = 'row_start'
    row_end = 'row_end'
    start_body = 'start_body'
    end_body = 'end_body'
    score_body = 'score_body'
    body = 'body'
    score = 'score'


class SignatureLabels(Enum):
    body = 'BODY'
    signature = 'SIGNATURE'


class SpacyTags(Enum):
    sentencizer = 'sentencizer'


class PipelineNames(Enum):
    segmentation_pipeline = 'segmentation_pipeline'
    signature_label = 'signature_label'
    sentence_parser_pipeline = 'sentence_parser_pipeline'
    ner_pipeline = 'ner_pipeline'
    defaule_pipeline_name = 'en'


class PipelineTasks(Enum):
    ner = 'ner'
    graph2text = 'graph2text'
    question_answering = 'question-answering'
    question_generation = 'question-generation'
