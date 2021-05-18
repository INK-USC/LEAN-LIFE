

# Steps to run the interactive documentation

1. `sh create_data_dirs.sh`
2. `pip install -r rqs.txt`
3. modify nltk source code:  in ```nltk/parse/chart.py```,  line 680, modify function ```parse```, change ```for edge in self.select(start=0, end=self._num_leaves,lhs=root):```  to  ```for edge in self.select(start=0, end=self._num_leaves):```
4. `python -m spacy download en_core_web_sm`
5. `uvicorn main:app --reload --port=9000`
6. http://127.0.0.1:9000/docs (for interactive docs)

Model Training Code will require CUDA 10.1, and if you're on linux you'll have to install it via
the pytorch site. The torch version is 1.7.0. For more help here, check [here](https://pytorch.org/get-started/previous-versions/#v170)

Additionally to set the GPUs available for model training, please change line 26 in `main.py`

# Managers

Once we have Data Models, Django gives us a database abstraction API that lets us create, retrieve, update, and delete objects. 
A model class represents database table, and an instance of that class represents a particular record in the database table.
We use a similar structure for our models. For more information, visit [this](https://docs.djangoproject.com/en/3.2/topics/db/managers/) page.

# Serializers

We use serailizer to convert the querysets and model instances to native Python datatypes that can be easily rendered into JSON, XML, etc.
We use different serializer like `NaturalLanguageExplanationSerializer`, `SingleUserAnnotatedDocumentExplanationsSerializer`, `SentimentAnalysisAnnotationSerializer`,  `RelationExtractionAnnotationHistorySerializer` etc. 

# Schema Definition 

## Lean Life 

### Params

```
"example": {

"experiment_name" : "test_experiment_1",

"dataset_name" : "test_dataset",

"dataset_size" : 3,

"project_type" : "Relation Extraction"

}
```
### Data

[This](https://github.com/INK-USC/LEAN-LIFE/blob/b43d5e64f377cfdcfcf6f81e28124ddcf40cdbb1/model_api/fast_api/json_schema.py#L129) is the sample data format for Lean Life.

## Training API 

### Params
```
"example": {

"stage" : "both",

"experiment_name" : "test_experiment_1",

"dataset_name" : "test_dataset",

"dataset_size" : 3,

"task" : "re",

"pre_train_build_data" : True,

"build_data" : True

}
```

### Data

[This](https://github.com/INK-USC/LEAN-LIFE/blob/b43d5e64f377cfdcfcf6f81e28124ddcf40cdbb1/model_api/fast_api/json_schema.py#L291) is the sample data format for the Training API.

## Evaluation API 

### Params


```
"params": {
    "experiment_name": "test_experiment_1",
    "dataset_name": "test_dataset",
    "train_dataset_size": 3,
    "task": "re",
    "none_label_key": "no_relation"
  }
```

### Data

[This](https://github.com/INK-USC/LEAN-LIFE/blob/b43d5e64f377cfdcfcf6f81e28124ddcf40cdbb1/model_api/fast_api/json_schema.py#L440) is the sample data format for the Evaluation API.

## Matching API 

### Strict Matching 

[This](https://github.com/INK-USC/LEAN-LIFE/blob/b43d5e64f377cfdcfcf6f81e28124ddcf40cdbb1/model_api/fast_api/json_schema.py#L377) is the sample data format for the Strict Matching API.


