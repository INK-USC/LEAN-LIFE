from fastapi import FastAPI
from fastapi import status
import torch
import util_functions as util_f
import json_schema as schema

app = FastAPI()

@app.post("/training/kickoff/lean-life/", status_code=status.HTTP_200_OK)
async def start_training_lean_life(lean_leaf_dataset: schema.LeanLifeDataset):
    util_f.kickstart_training(lean_leaf_dataset)
    return

@app.post("/training/kickoff/api/", status_code=status.HTTP_200_OK)
async def start_training_lean_life(api_dataset: schema.ExplanationDataset):
    util_f.kickstart_training(api_dataset, lean_life=False)
    return 

@app.get("/download/{file_path:path}")
async def get_trained_model(file_path: str):
    state_dict = torch.load(file_path, map_location="cpu")
    return state_dict