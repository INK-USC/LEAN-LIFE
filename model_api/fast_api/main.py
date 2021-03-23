from fastapi import FastAPI
from fastapi import status
import os
import torch
import sys
sys.path.append(".")
sys.path.append("../model_training/")
import fast_api_util_functions as util_f
import json_schema as schema
from internal_api.internal_main import train_next_framework_lean_life, train_next_framework, apply_strict_matching, evaluate_next

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
app = FastAPI()

@app.post("/training/next/lean-life/", status_code=status.HTTP_200_OK, response_model=schema.SavePathOutput)
async def start_next_training_lean_life(lean_life_payload: schema.LeanLifePayload):
    params = lean_life_payload.params
    lean_life_data = lean_life_payload.lean_life_data
    prepped_data = util_f.prepare_next_data(lean_life_data, task=params["project_type"])
    label_space, unlabeled_docs, explanation_triples, ner_label_space = prepped_data
    if len(ner_label_space) == 0:
        ner_label_space = None
    if len(unlabeled_docs) == 0:
        unlabeled_docs = None
    if len(explanation_triples) == 0:
        explanation_triples = None
    return schema.SavePathOutput(train_next_framework_lean_life(params, label_space,
                                                                unlabeled_docs,
                                                                explanation_triples,
                                                                ner_label_space))
    
@app.post("/training/next/api/", status_code=status.HTTP_200_OK, response_model=schema.SavePathOutput)
async def start_next_training_api(api_payload: schema.ExplanationTrainingPayload):
    prepped_data = util_f.prepare_next_data(api_payload, lean_life=False)
    label_space, unlabeled_docs, explanation_data, ner_label_space = prepped_data
    if len(ner_label_space) == 0:
        ner_label_space = None
    if len(unlabeled_docs) == 0:
        unlabeled_docs = None
    if len(explanation_triples) == 0:
        explanation_triples = None
    return schema.SavePathOutput(train_next_framework(params, label_space, unlabeled_docs,
                                                      explanation_triples, ner_label_space))

@app.post("/training/next/eval", status_code=status.HTTP_200_OK, response_model=schema.NextEvalDataOutput)
async def eval_next_clf(api_payload: schema.EvalNextClfPayload):
    params = api_payload.params
    params["label_map"] = api_payload.label_space
    params["eval_data"] = api_payload.eval_data
    return schema.NextEvalDataOutput(evaluate_next(params))

@app.get("/download/{file_path:path}")
async def get_trained_model(file_path: str):
    state_dict = torch.load(file_path, map_location="cpu")
    for key in state_dict:
        state_dict[key] = state_dict[key].numpy().tolist()
    return json.loads(json.dumps(state_dict))

@app.post("/other/next/match", status_code=status.HTTP_200_OK, response_model=schema.MatchedDataOutput)
async def strict_match_data(api_payload: schema.StrictMatchPayload):
    return schema.MatchedDataOutput(apply_strict_matching(api_payload))

