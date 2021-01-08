from fastapi import FastAPI
from fastapi import status
import util_functions as util_f

app = FastAPI()

@app.post("/training/kickoff/lean-life/", status_code=status.HTTP_204_NO_CONTENT)
async def start_training_lean_life(lean_leaf_dataset: UploadFile = File(...)):
    util_f.kickstart_training(lean_leaf_dataset.file)
    return

@app.post("/training/kickoff/api/", status_code=status.HTTP_204_NO_CONTENT)
async def start_training_lean_life(api_dataset: UploadFile = File(...)):
    util_f.kickstart_training(api_dataset.file, lean_life=False)
    return

@app.get("/download/")
async def get_trained_model(file_path: str):
    with open(file_path) as f:
        return {"model_file" : f}