from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Union

class Label(BaseModel):
    id : int
    text : str
    user_provided : bool

class AnnotatedDoc(BaseModel):
    text : str
    annotations : List[Any]
    explanations: List[Any]

class PlainReDoc(BaseModel):
    text : str
    annotations : List[Any]

class UnlabeledDoc(BaseModel):
    text : str

class LeanLifeDataset(BaseModel):
    project_type : str
    model_name : str
    label_space : List[Label]
    annotated : List[AnnotatedDoc]
    unlabeled : List[Union[PlainReDoc, UnlabeledDoc]]

class ExplanationDataset(BaseModel):
    training_pairs : List[Tuple[str, str]]
    explanation_triples : Dict[str, Union[List[str], str]]
    label_space : List[str]
    unlabeled_text : List[str]
