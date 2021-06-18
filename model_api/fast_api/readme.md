Implementation details about NeXT Framework can be found [here](../model_training/next_framework/readme.md)

# Table of contents

1. [Steps to run the FAST api](#steps-to-run-the-fast-api)
2. [Schema Definition](#schema-definition)

# Steps to run the FAST api

1. Ensure your `leanlife` environment is activated
2. Execute `sh create_data_dirs.sh`
3. Execute `pip install -r rqs.txt`
4. modify nltk source code: in `nltk/parse/chart.py`, line 680, modify function `parse`,
   change `for edge in self.select(start=0, end=self._num_leaves,lhs=root):`
   to `for edge in self.select(start=0, end=self._num_leaves):`
5. `python -m spacy download en_core_web_sm`
6. `uvicorn main:app --reload --port=9000`
7. The API is now exposed on port 9000. Swagger interactive API documentation can be
   found [here](http://127.0.0.1:9000/docs)

**CUDA**:

Model Training Code will require CUDA 10.1, and if you're on linux you'll have to install it via the pytorch site. The
torch version is 1.7.0. For more help here, check [here](https://pytorch.org/get-started/previous-versions/#v170)

Additionally to set the GPUs available for model training, please change line 28 in `main.py`

# Schema Definition

The schema definition can be found in [JSON schema file](json_schema.py). And can also be accessed via Swagger
interactive API documentation.
