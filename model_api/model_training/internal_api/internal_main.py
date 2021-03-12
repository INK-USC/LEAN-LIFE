from os import path
import warnings
import sys
sys.path.appned(".")
from internal_api.pipelines import pre_train_find_module_pipeline, train_next_bilstm_pipeline, strict_match_pipeline

class GeneralPrintError(Excpetion):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def train_next_framework_lean_life(params, label_space, unlabeled_docs=None, explanation_triples=None, ner_label_space=None):
    explanation_data = None
    
    # Sending data through LEAN-LIFE is more controlled
    # If data is sent, train full pipeline
    # If data is not sent, just tune downstream classifier
    if explanation_triples != None:
        explanation_data = []
        # for now we are just "AND"ing all explanations for an annotation
        for key in explanation_triples:
            triple_list =
    
        if unlabeled_docs != None:
            params["stage"] = "both"
            params["pre_train_build_data"] = True
            params["build_data"] = True
        else:
            error_msg = "No Training Data Sent"
            raise GeneralPrintError(error_msg)
    else:
        params["stage"] = "clf"
    
    if params["project_type"] == "Sentiment Analysis":
        params["task"] = "sa"
    elif params["project_type"] == "Relation Extraction":
        params["task"] = "re"

    train_next_framework(params, unlabeled_data, explanation_data, label_space, ner_label_space)

def train_next_framework(params, label_space, unlabeled_data=None, explanation_data=None, ner_label_space=None): 
    required_keys = ["stage", "experiment_name", "build_data", "pre_train_build_data", "dataset_name", "dataset_size", "task"]
    for key in required_keys:
        if key not in params:
            error_msg = "{} not found in params".format(key)
            raise GeneralPrintError(error_msg)
    
    stage = params["stage"] # inferred for lean_life
    experiment_name = params["experiment_name"] 
    build_data = params["build_data"]
    build_pretrain_data = params["pre_train_build_data"]
    task = params["task"]

    if stage not in ["both", "clf", "find"]:
        error_msg = "stage must be one of `both`, `clf` or `find`"
        raise GeneralPrintError(error_msg)

    if task not in ["re", "sa"]:
        error_msg = "Task must be one of re or sa, sa is really just text classification"
        raise GeneralPrintError(error_msg)
    
    if stage == "clf":
        find_module_path =  "../model_training/next_framework/data/saved_models/Find-Module-pt_{}.p".format(experiment_name)
        if not path.isfile(find_module_path):
            error_msg = "FIND Module must be created first, please send stage in training data and/or set `stage` to `both`"
            raise GeneralPrintError(error_msg)
    
    if stage == "both" or build_data or build_pretrain_data:
        if unlabeled_data == None or explanation_data == None:
            error_msg = "In order to build datasets data must be provided"
            raise GeneralPrintError(error_msg)
        
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
    
    if stage == "find":
        pre_train_find_module_pipeline(params)
    else:
        train_next_bilstm_pipeline(params)