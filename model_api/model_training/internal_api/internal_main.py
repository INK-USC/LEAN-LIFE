"""
    Intermediary layer between API and pipelines. Fast API functions pass data through to this layer
    to do final validation checks before data is packaged in a single payload and sent over to a pipeline.

    Handling of async should be done here, a useful resource might be this entry from Fast API on async
    https://fastapi.tiangolo.com/async/

    Ultimately I think this code should be coverted into a worker class that reads from a queue and processes
    jobs.
"""
from os import path
import pathlib
import sys
import warnings
from fastapi import HTTPException
PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
sys.path.append(PATH_TO_PARENT + "../")
sys.path.append(PATH_TO_PARENT + "../../")

from internal_api.next_pipelines import pre_train_find_module_pipeline, train_next_bilstm_pipeline, strict_match_data, \
        soft_match_data, evaluate_next_clf_pipeline, hard_match_bilstm_pipeline, predict_next_clf_pipeline
from internal_api.standard_pipeline import standard_pipeline, evaluate_standard_clf_pipeline, predict_standard_clf_pipeline
from fast_api.fast_api_util_functions import lean_life_error_catcher_decorator
from internal_api.trigger_pipelines import standard_ner_pipeline, evaluate_standard_ner_pipeline, \
        predict_standard_ner_pipeline


def _remove_trailing_period(string):
    if string[len(string)-1] == ".":
        return string[:len(string)-1]
    return string


@lean_life_error_catcher_decorator
def train_next_framework_lean_life(params, label_space, unlabeled_docs=None, explanation_triples=None, ner_label_space=None):
    """
        Function that transforms converted LEAN-LIFE datastructures into a more standard form for training purposes

        Is called by the `start_next_training_lean_life` api function in "main.py"

        This function really just calls the `train_next_framework` after some custom LEAN-LIFE dataprep
    """
    explanation_data = None

    # If data is sent, train full pipeline
    # If data is not sent, just tune downstream classifier
    if explanation_triples != None:
        explanation_data = []
        # for now we are just "AND"ing all explanations for an annotation
        for entry in explanation_triples:
            explanations = [_remove_trailing_period(explanation) for explanation in entry["explanation"]]
            explanation = " and ".join(explanations)
            explanation_data.append({"text" : entry["text"], "label" : entry["label"], "explanation" : explanation})

        if unlabeled_docs != None:
            params["stage"] = "both"
            params["pre_train_build_data"] = True
            params["build_data"] = True
        else:
            error_msg = "No Training Data Sent"
            raise HTTPException(status_code=500, detail=error_msg)
    else:
        params["stage"] = "clf"
        params["pre_train_build_data"] = False
        params["build_data"] = False
    
    if params["project_type"] == "Sentiment Analysis":
        params["task"] = "sa"
    elif params["project_type"] == "Relation Extraction":
        params["task"] = "re"
    elif params["project_type"] == "Named Entity Recognition":
        params["task"] = "ner"

    params["leanlife"] = True

    return train_next_framework(params, label_space, unlabeled_docs, explanation_data, ner_label_space)

def train_next_framework(params, label_space, unlabeled_data=None, explanation_data=None, ner_label_space=None):
    """
        Apart from data just be presented in the right format, we do some additional data validation checks before
        sending the data through our next training framework. Checks are listed below.
    """
    # Ensure user sends at least these parameter settings, should already be checked by validators on receiving request
    # so more of a double check
    required_keys = ["stage", "experiment_name", "build_data", "pre_train_build_data", "dataset_name", "dataset_size", "task"]
    for key in required_keys:
        if key not in params:
            error_msg = "{} not found in params".format(key)
            raise HTTPException(status_code=500, detail=error_msg)
    
    # this key is only set in `train_next_framework_lean_life()`, so if its not there its not LEAN-LIFE data
    if "leanlife" not in params:
        params["leanlife"] = False

    stage = params["stage"] # inferred for lean_life
    experiment_name = params["experiment_name"] 
    build_data = params["build_data"]
    build_pretrain_data = params["pre_train_build_data"]
    task = params["task"]

    # should already be checked by validators on receiving request (double check)
    if stage not in ["both", "clf", "find"]:
        error_msg = "stage must be one of `both`, `clf` or `find`"
        raise HTTPException(status_code=500, detail=error_msg)
    
    # should already be checked by validators on receiving request (double check)
    if task not in ["re", "sa"]:
        error_msg = "Task must be one of re or sa, sa is really just text classification"
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Can't train downstream classifier without FIND module being present
    if stage == "clf":
        find_module_path =  PATH_TO_PARENT + "../next_framework/data/saved_models/Find-Module-pt_{}.p".format(experiment_name)
        if not path.isfile(find_module_path):
            error_msg = "FIND Module must be created first, please send appropraite training data and/or set `stage` to `both`"
            raise HTTPException(status_code=500, detail=error_msg)
    
    # makes sure if training is to be done that data is provided
    if stage == "both" or build_data or build_pretrain_data:
        if unlabeled_data == None or explanation_data == None:
            error_msg = "In order to build datasets data must be provided"
            raise HTTPException(status_code=500, detail=error_msg)
        
        params["training_data"] = unlabeled_data
        params["explanation_data"] = explanation_data
    
    # provides a warning of the lack of ner_labels if the task is re
    if task == "re": 
        if ner_label_space == None:
            warning_msg = """
                Knowledge of the NER label space is generally needed for relation extraction.
                We will used SPACY's NER labels as default, but please provide a custom label
                space for better performance
            """
            warnings.warn(warning_msg)
        else:
            params["ner_labels"] = ner_label_space 

    params["label_map"] = label_space
    
    if "soft_match" not in params or params["soft_match"] is None:
        params["soft_match"] = True

    # trains only find_module
    if stage == "find":
        return pre_train_find_module_pipeline(params)
    else: # clf and both both trigger this
        if params["soft_match"] == True:
            return train_next_bilstm_pipeline(params)
        else:
            return hard_match_bilstm_pipeline(params)

def apply_strict_matching(payload):
    """Just passes along the payload"""
    return strict_match_data(payload)

def apply_soft_matching(params, label_space, unlabeled_docs=None, explanation_triples=None, ner_label_space=None):
    """Just passes along the payload"""
    explanation_data = None

    # If data is sent, train full pipeline
    # If data is not sent, just tune downstream classifier
    if explanation_triples != None:
        explanation_data = []
        # for now we are just "AND"ing all explanations for an annotation
        for entry in explanation_triples:
            explanations = [_remove_trailing_period(explanation) for explanation in entry["explanation"]]
            explanation = " and ".join(explanations)
            explanation_data.append({"text" : entry["text"], "label" : entry["label"], "explanation" : explanation})

        if unlabeled_docs != None:
            params["stage"] = "both"
            params["pre_train_build_data"] = True
            params["build_data"] = True
        else:
            error_msg = "No Training Data Sent"
            raise HTTPException(status_code=500, detail=error_msg)
    else:
        params["stage"] = "clf"
        params["pre_train_build_data"] = False
        params["build_data"] = False
    
    if params["project_type"] == "Sentiment Analysis":
        params["task"] = "sa"
    elif params["project_type"] == "Relation Extraction":
        params["task"] = "re"
        
    
    # Ensure user sends at least these parameter settings, should already be checked by validators on receiving request
    # so more of a double check
    required_keys = ["stage", "experiment_name", "build_data", "pre_train_build_data", "dataset_name", "dataset_size", "task"]
    for key in required_keys:
        if key not in params:
            error_msg = "{} not found in params".format(key)
            raise HTTPException(status_code=500, detail=error_msg)
    
    # this key is only set in `train_next_framework_lean_life()`, so if its not there its not LEAN-LIFE data
    if "leanlife" not in params:
        params["leanlife"] = False

    stage = params["stage"] # inferred for lean_life
    experiment_name = params["experiment_name"] 
    build_data = params["build_data"]
    build_pretrain_data = params["pre_train_build_data"]
    task = params["task"]

    # should already be checked by validators on receiving request (double check)
    if stage not in ["both", "clf", "find"]:
        error_msg = "stage must be one of `both`, `clf` or `find`"
        raise HTTPException(status_code=500, detail=error_msg)
    
    # should already be checked by validators on receiving request (double check)
    if task not in ["re", "sa"]:
        error_msg = "Task must be one of re or sa, sa is really just text classification"
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Can't train downstream classifier without FIND module being present
    if stage == "clf":
        find_module_path =  PATH_TO_PARENT + "../next_framework/data/saved_models/Find-Module-pt_{}.p".format(experiment_name)
        if not path.isfile(find_module_path):
            error_msg = "FIND Module must be created first, please send appropraite training data and/or set `stage` to `both`"
            raise HTTPException(status_code=500, detail=error_msg)
    
    # makes sure if training is to be done that data is provided
    if stage == "both" or build_data or build_pretrain_data:
        if unlabeled_docs == None or explanation_data == None:
            error_msg = "In order to build datasets data must be provided"
            raise HTTPException(status_code=500, detail=error_msg)
        
        params["training_data"] = unlabeled_docs
        params["explanation_data"] = explanation_data
    
    # provides a warning of the lack of ner_labels if the task is re
    if task == "re": 
        if ner_label_space == None:
            warning_msg = """
                Knowledge of the NER label space is generally needed for relation extraction.
                We will used SPACY's NER labels as default, but please provide a custom label
                space for better performance
            """
            warnings.warn(warning_msg)
        else:
            params["ner_labels"] = ner_label_space 

    params["label_map"] = label_space

    return soft_match_data(params)

def evaluate_next(payload, ner_label_space=None):
    """Just passes along the payload"""
    task = payload["task"]
    if task == "re":
        if ner_label_space == None:
            warning_msg = """
                Knowledge of the NER label space is generally needed for relation extraction.
                We will used SPACY's NER labels as default, but please provide a custom label
                space for better performance
            """
            warnings.warn(warning_msg)
        else:
            payload["ner_labels"] = ner_label_space
    return evaluate_next_clf_pipeline(payload)


def train_standard_pipeline(params, label_space, labeled_data=None, dev_data=None, ner_label_space=None):
    # Ensure user sends at least these parameter settings, should already be checked by validators on receiving request
    # so more of a double check
    required_keys = ["experiment_name", "build_data", "dataset_name", "task"]
    for key in required_keys:
        if key not in params:
            error_msg = "{} not found in params".format(key)
            raise HTTPException(status_code=500, detail=error_msg)

    # this key is only set in `train_next_framework_lean_life()`, so if its not there its not LEAN-LIFE data
    if "leanlife" not in params:
        params["leanlife"] = False

    build_data = params["build_data"]
    task = params["task"]

    if build_data:
        if labeled_data is None or len(labeled_data) == 0:
            error_msg = "labeled_data is required for training"
            raise HTTPException(status_code=500, detail=error_msg)

        params["labeled_data"] = labeled_data
        params["dev_data"] = dev_data

    # provides a warning of the lack of ner_labels if the task is re
    if task == "re":
        if ner_label_space is None:
            warning_msg = """
                Knowledge of the NER label space is generally needed for relation extraction.
                We will used SPACY's NER labels as default, but please provide a custom label
                space for better performance
            """
            warnings.warn(warning_msg)
        else:
            params["ner_labels"] = ner_label_space

    params["label_map"] = label_space

    return standard_pipeline(params)


@lean_life_error_catcher_decorator
def train_standard_lean_life(params, label_space, labeled_data=None, dev_data=None, ner_label_space=None):
    """
        Function that transforms converted LEAN-LIFE datastructures into a more standard form for training purposes

        Is called by the `start_standard_training_lean_life` api function in "main.py"

        This function really just calls the `train_standard_pipeline` after some custom LEAN-LIFE dataprep
    """
    # If data is sent, train full pipeline
    # If data is not sent, just tune downstream classifier
    if labeled_data is not None:
        params["build_data"] = True
    else:
        params["build_data"] = False

    if params["project_type"] == "Sentiment Analysis":
        params["task"] = "sa"
    elif params["project_type"] == "Relation Extraction":
        params["task"] = "re"
    elif params["project_type"] == "Named Entity Recognition":
        params["task"] = "ner"

    params["leanlife"] = True

    return train_standard_pipeline(params, label_space, labeled_data, dev_data, ner_label_space)


def evaluate_standard(params, ner_label_space):
    """Just passes along the payload"""
    # provides a warning of the lack of ner_labels if the task is re
    task = params["task"]
    if task == "re":
        if ner_label_space == None:
            warning_msg = """
                Knowledge of the NER label space is generally needed for relation extraction.
                We will used SPACY's NER labels as default, but please provide a custom label
                space for better performance
            """
            warnings.warn(warning_msg)
        else:
            params["ner_labels"] = ner_label_space
    return evaluate_standard_clf_pipeline(params)


def predict_next(payload, ner_label_space):
    """Just passes along the payload"""
    task = payload["task"]
    if task == "re":
        if ner_label_space == None:
            warning_msg = """
                Knowledge of the NER label space is generally needed for relation extraction.
                We will used SPACY's NER labels as default, but please provide a custom label
                space for better performance
            """
            warnings.warn(warning_msg)
        else:
            payload["ner_labels"] = ner_label_space
    return predict_next_clf_pipeline(payload)


def predict_standard(params, ner_label_space):
    """Just passes along the payload"""
    # provides a warning of the lack of ner_labels if the task is re
    task = params["task"]
    if task == "re":
        if ner_label_space == None:
            warning_msg = """
                Knowledge of the NER label space is generally needed for relation extraction.
                We will used SPACY's NER labels as default, but please provide a custom label
                space for better performance
            """
            warnings.warn(warning_msg)
        else:
            params["ner_labels"] = ner_label_space
    return predict_standard_clf_pipeline(params)


@lean_life_error_catcher_decorator
def train_standard_ner_lean_life(params, labeled_data=None, dev_data=None):
    """
        Function that transforms converted LEAN-LIFE datastructures into a more standard form for training purposes

        Is called by the `start_standard_training_ner_lean_life` api function in "main.py"

        This function really just calls the `train_standard_ner_pipeline` after some custom LEAN-LIFE dataprep
    """
    # If data is sent, train full pipeline
    # If data is not sent, just tune downstream classifier
    if labeled_data is not None:
        params["build_data"] = True
    else:
        params["build_data"] = False

    if params["project_type"] == "Sentiment Analysis":
        params["task"] = "sa"
    elif params["project_type"] == "Relation Extraction":
        params["task"] = "re"
    elif params["project_type"] == "Named Entity Recognition":
        params["task"] = "ner"

    params["leanlife"] = True

    return train_standard_ner_pipeline(params, labeled_data, dev_data)


def train_standard_ner_pipeline(params, labeled_data=None, dev_data=None):
    # Ensure user sends at least these parameter settings, should already be checked by validators on receiving request
    # so more of a double check
    required_keys = ["experiment_name", "build_data", "dataset_name"]
    for key in required_keys:
        if key not in params:
            error_msg = "{} not found in params".format(key)
            raise HTTPException(status_code=500, detail=error_msg)

    # this key is only set in `train_standard_ner_pipeline_lean_life()`, so if its not there its not LEAN-LIFE data
    if "leanlife" not in params:
        params["leanlife"] = False

    build_data = params["build_data"]

    if build_data:
        if labeled_data is None or len(labeled_data) == 0:
            error_msg = "labeled_data is required for training"
            raise HTTPException(status_code=500, detail=error_msg)

        params["labeled_data"] = labeled_data
        params["dev_data"] = dev_data

    return standard_ner_pipeline(params)


def evaluate_standard_ner(params):
    """Just passes along the payload"""
    return evaluate_standard_ner_pipeline(params)


def predict_standard_ner(params):
    """Just passes along the payload"""
    return predict_standard_ner_pipeline(params)
