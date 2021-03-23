from os import path
import warnings
import sys
import pathlib
PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
# sys.path.appned(".")
sys.path.append(PATH_TO_PARENT)
from internal_api.pipelines import pre_train_find_module_pipeline, train_next_bilstm_pipeline, strict_match_pipeline, evaluate_next_clf
from fastapi import HTTPException

def _remove_trailing_period(string):
    if string[len(string)-1] == ".":
        return string[:len(string)-1]
    return string

def train_next_framework_lean_life(params, label_space, unlabeled_docs=None, explanation_triples=None, ner_label_space=None):
    explanation_data = None
    # Sending data through LEAN-LIFE is more controlled
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
    
    if params["project_type"] == "Sentiment Analysis":
        params["task"] = "sa"
    elif params["project_type"] == "Relation Extraction":
        params["task"] = "re"

    params["leanlife"] = True

    return train_next_framework(params, label_space, unlabeled_data, explanation_data, ner_label_space)

def train_next_framework(params, label_space, unlabeled_data=None, explanation_data=None, ner_label_space=None): 
    required_keys = ["stage", "experiment_name", "build_data", "pre_train_build_data", "dataset_name", "dataset_size", "task"]
    for key in required_keys:
        if key not in params:
            error_msg = "{} not found in params".format(key)
            raise HTTPException(status_code=500, detail=error_msg)
    
    if "leanlife" not in params:
        params["leanlife"] = False

    stage = params["stage"] # inferred for lean_life
    experiment_name = params["experiment_name"] 
    build_data = params["build_data"]
    build_pretrain_data = params["pre_train_build_data"]
    task = params["task"]

    if stage not in ["both", "clf", "find"]:
        error_msg = "stage must be one of `both`, `clf` or `find`"
        raise HTTPException(status_code=500, detail=error_msg)

    if task not in ["re", "sa"]:
        error_msg = "Task must be one of re or sa, sa is really just text classification"
        raise HTTPException(status_code=500, detail=error_msg)
    
    if stage == "clf":
        find_module_path =  PATH_TO_PARENT + "../next_framework/data/saved_models/Find-Module-pt_{}.p".format(experiment_name)
        if not path.isfile(find_module_path):
            error_msg = "FIND Module must be created first, please send stage in training data and/or set `stage` to `both`"
            raise HTTPException(status_code=500, detail=error_msg)
    
    if stage == "both" or build_data or build_pretrain_data:
        if unlabeled_data == None or explanation_data == None:
            error_msg = "In order to build datasets data must be provided"
            raise HTTPException(status_code=500, detail=error_msg)
        
        params["training_data"] = unlabeled_data
        params["explanation_data"] = explanation_data
    
    if task == "re": 
        if ner_labels == None:
            warning_msg = """
                Knowledge of the NER label space is generally needed for relation extraction.
                We will used SPACY's NER labels as default, but please provide a custom label
                space for better performance
            """
            warnings.warn(warning_msg)
        else:
            params["ner_labels"] = ner_labels 

    params["label_map"] = label_space
    
    if stage == "find":
        return pre_train_find_module_pipeline(params)
    else:
        return train_next_bilstm_pipeline(params)

def apply_strict_matching(payload):
    return strict_match_pipeline(payload)

def evaluate_next(payload):
    return evaluate_next_clf(payload)