from fastapi import FastAPI
from fastapi import status
import torch
import util_functions as util_f
import json_schema as schema
import sys
sys.path.append("../model_training/")
from internal_api.internal_main import train_next_framework_lean_life, train_next_framework
app = FastAPI()

@app.post("/training/next/kickoff/lean-life/", status_code=status.HTTP_200_OK)
async def start_next_training_lean_life(lean_life_payload: schema.LeanLifePayload):
    params = lean_life_payload.params
    lean_life_data = lean_life_payload.lean_life_data
    prepped_data = util_f.prepare_next_data(lean_life_data)
    label_space, unlabeled_docs, explanation_triples, ner_label_space = prepped_data
    if len(ner_label_space):
        ner_label_space = None
    if len(unlabeled_docs):
        unlabeled_docs = None
    if len(explanation_triples):
        explanation_triples = None
    train_next_framework_lean_life(params, label_space, unlabeled_docs, explanation_triples, ner_label_space)
    
@app.post("/training/next/kickoff/api/", status_code=status.HTTP_200_OK)
async def start_next_training_lean_life(api_dataset: schema.ExplanationDataset):
    prepped_data = util_f.prepare_next_data(api_dataset, lean_life=False)
    label_space, unlabeled_docs, explanation_data, ner_label_space = prepped_data
    if len(ner_label_space):
        ner_label_space = None
    if len(unlabeled_docs):
        unlabeled_docs = None
    if len(explanation_triples):
        explanation_triples = None
    train_next_framework(params, label_space, unlabeled_docs, explanation_triples, ner_label_space)

@app.get("/download/{file_path:path}")
async def get_trained_model(file_path: str):
    state_dict = torch.load(file_path, map_location="cpu")
    return state_dict