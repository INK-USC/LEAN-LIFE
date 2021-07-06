This directory contains all the code related to model training and exposing it using [FastAPI](https://fastapi.tiangolo.com/)

## Directory structure

1. `fast_api/`

   - Entrypoint for all training routes/pipelines
   - A thin wrapper over pipelines in `model_training` to expose them as routes and handle processing of input data.

2. `model_training/`

   Consists of model training pipelines, modules like `next_framework` and more to join.

   - `internal_api/`

     - `defaults.py`: default values of hyper parameters
     - `internal_main.py`: entrypoint to pipelines which is accessed by `fast_api`
     - `next_pipelines.py`: consists of training pipelines pertaining to the `next_framework`
     - `standard_pipeline.py`: consists standard training pipeline code

   - `next_framework/`

     Consists of model training code and helpers required for Natural Language Explanation based training. More details in its [readme](next_framework/readme.md)
