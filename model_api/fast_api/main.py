"""
    Fast API Endpoint file

    Designed to be "dumb", does some data validation and data transformation, but really passes off
    the heavy lifting further down the pipeline.

    To run this model-training api, run the following command (assuming all packages are installed): 
        `uvicorn main:app --reload --port=9000`
    
    To access interactive docs and dummy test data navigate to http://127.0.0.1:9000/docs after running the
    above command
"""
import json
import os
import sys
import torch
from fastapi import FastAPI
from fastapi import status
sys.path.append(".")
sys.path.append("../model_training/")
import fast_api_util_functions as util_f
import json_schema as schema
from internal_api.internal_main import train_next_framework_lean_life, train_next_framework, apply_strict_matching, evaluate_next

# We don't have a sophisticated CUDA Management policy, so please make needed changes to fit your needs
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
app = FastAPI()

@app.post("/training/next/lean-life/", status_code=status.HTTP_200_OK, response_model=schema.SavePathOutput)
async def start_next_training_lean_life(lean_life_payload: schema.LeanLifePayload):
    """
        Endpoint hit by annotation tool's django api
    """
    params = lean_life_payload.params
    lean_life_data = lean_life_payload.lean_life_data
    prepped_data = util_f.prepare_next_data(lean_life_data, task=params.project_type)
    label_space, unlabeled_docs, explanation_triples, ner_label_space = prepped_data
    if len(ner_label_space) == 0:
        ner_label_space = None
    if len(unlabeled_docs) == 0:
        unlabeled_docs = None
    if len(explanation_triples) == 0:
        explanation_triples = None
    return schema.SavePathOutput(train_next_framework_lean_life(params.__dict__, label_space,
                                                                unlabeled_docs,
                                                                explanation_triples,
                                                                ner_label_space))
    
@app.post("/training/next/api/", status_code=status.HTTP_200_OK, response_model=schema.SavePathOutput)
async def start_next_training_api(api_payload: schema.ExplanationTrainingPayload):
    """
        Endpoint used to kick off training of a classifier via the next framework of learning from
        explanations. Please refer to the docs or `json_schema.py` to understand both the supported 
        and required paramaters.
    """
    prepped_data = util_f.prepare_next_data(api_payload, lean_life=False)
    label_space, unlabeled_docs, explanation_triples, ner_label_space = prepped_data
    if len(ner_label_space) == 0:
        ner_label_space = None
    if len(unlabeled_docs) == 0:
        unlabeled_docs = None
    if len(explanation_triples) == 0:
        explanation_triples = None
    return schema.SavePathOutput(train_next_framework(params.__dict__, label_space, unlabeled_docs,
                                                      explanation_triples, ner_label_space))

@app.post("/training/next/eval", status_code=status.HTTP_200_OK, response_model=schema.NextEvalDataOutput)
async def eval_next_clf(api_payload: schema.EvalNextClfPayload):
    """
        Endpoint used to evaluate a classifier trained via the next framework. Please refer to the docs or
        `json_schema.py` to understand both the supported and required paramaters.
    """
    params = api_payload.params.__dict__
    params["label_map"] = api_payload.label_space
    params["eval_data"] = api_payload.eval_data
    return schema.NextEvalDataOutput(evaluate_next(params))

@app.get("/download/{file_path:path}")
async def get_trained_model(file_path: str):
    """
        Endpoint used to load a saved model's weight and send them back to requester

        Model weights are the model's state_dict, but instead of tensors we save the weights
        in Python Lists so that we can serialize the state_dict. To get back the original 
        state_dict, per key convert the list back to a PyTorch Tensor.
    """
    state_dict = torch.load(file_path, map_location="cpu")
    for key in state_dict:
        state_dict[key] = state_dict[key].numpy().tolist()
    return json.loads(json.dumps(state_dict))

@app.post("/other/next/match", status_code=status.HTTP_200_OK, response_model=schema.MatchedDataOutput)
async def strict_match_data(api_payload: schema.StrictMatchPayload):
    """
        Endpoint that converts explanations into strict labeling functions and labels
        a pool of unlabeled sentences. Please refer to the docs or `json_schema.py` to
        understand the required paramaters.
    """
    return schema.MatchedDataOutput(apply_strict_matching(api_payload.__dict__))

