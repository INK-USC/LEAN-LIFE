from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Union, Optional

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

class LeanLifeData(BaseModel):
    label_space : List[Label]
    annotated : Optional(List[AnnotatedDoc])
    unlabeled : Optional(List[Union[PlainReDoc, UnlabeledDoc]])

class LeanLifePayload(BaseModel):
    lean_life_data : LeanLifeData
    params : Dict[str, str]

class ExplanationDataset(BaseModel):
    # training_pairs : List[Tuple[str, str]]
    explanation_triples : Optional(List[Dict[str, str]])
    label_space : List[str]
    unlabeled_text : List[str]
    params : Dict[str, str]
    ner_label_space : Optional(List[str])
