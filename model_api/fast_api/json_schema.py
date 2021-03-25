from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Union, Optional
from typing_extensions import Literal

class SavePathOutput(BaseModel):
    save_path : str

class LeanLifeParams(BaseModel):
    experiment_name : str
    dataset_name : str
    dataset_size : int
    project_type : str
    match_batch_size : Optional[str]
    unlabeled_batch_size : Optional[str]
    learning_rate : Optional[float]
    epochs : Optional[int]
    embeddings : Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                  'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                  'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim : Optional[int]
    gamma : Optional[float]
    hidden_dim : Optional[int]
    random_state : Optional[int]
    load_model : Optional[bool]
    start_epoch : Optional[int]
    pre_train_hidden_dim : Optional[int]
    pre_train_training_size : Optional[int]

    class Config:
        schema_extra = {
            "example": {
                "experiment_name" : "test_experiment_1",
                "dataset_name" : "test_dataset",
                "dataset_size" : 3,
                "project_type" : "Relation Extraction"
            }
        }

class TrainingApiParams(BaseModel):
    stage : Literal["both", "clf", "find"]
    experiment_name : str
    dataset_name : str
    dataset_size : int
    task : Literal['sa', 're']
    pre_train_build_data : bool
    build_data : bool
    
    embeddings : Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                  'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                  'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim : Optional[int]
    custom_vocab_tokens : Optional[List[str]]
    relation_ner_types : Optional[Dict[str, Tuple[str, str]]]

    pre_train_batch_size : Optional[int]
    pre_train_eval_batch_size : Optional[int]
    pre_train_learning_rate : Optional[float]
    pre_train_epochs : Optional[int]
    pre_train_emb_dim : Optional[int]
    pre_train_hidden_dim : Optional[int]
    pre_train_training_size : Optional[int]
    pre_train_random_state : Optional[int]
    pre_train_gamma : Optional[float]
    pre_train_load_model : Optional[bool]
    pre_train_start_epoch : Optional[int]

    match_batch_size : Optional[int]
    unlabeled_batch_size : Optional[int]
    eval_batch_size : Optional[int]
    learning_rate : Optional[float]
    epochs : Optional[int]
    gamma : Optional[float]
    hidden_dim : Optional[int]
    random_state : Optional[int]
    none_label_key : Optional[str]
    load_model : Optional[bool]
    start_epoch : Optional[int]
    eval_data : Optional[List[Tuple[str, str]]]

    class Config:
        schema_extra = {
            "example": {
                "stage" : "both",
                "experiment_name" : "test_experiment_1",
                "dataset_name" : "test_dataset",
                "dataset_size" : 3,
                "task" : "re",
                "pre_train_build_data" : True,
                "build_data" : True
            }
        }

class EvalApiParams(BaseModel):
    experiment_name : str
    dataset_name : str
    train_dataset_size : int
    embeddings : Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                  'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                  'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim : Optional[int]
    custom_vocab_tokens : Optional[List[str]]
    hidden_dim : Optional[int]
    none_label_key : Optional[str]
    pre_train_training_size : Optional[int]
    eval_batch_size : Optional[int]

class Label(BaseModel):
    id : int
    text : str
    user_provided : bool

class AnnotatedDoc(BaseModel):
    text : str
    annotations : List[Dict[str, Union[str, bool, int]]]
    explanations: List[Dict[str, Union[str, int]]]

class PlainReDoc(BaseModel):
    text : str
    annotations : List[Dict[str, Union[str, bool, int]]]

class UnlabeledDoc(BaseModel):
    text : str

class LeanLifeData(BaseModel):
    label_space : List[Label]
    annotated : Optional[List[AnnotatedDoc]]
    unlabeled : Optional[List[Union[PlainReDoc, UnlabeledDoc]]]

    class Config:
        schema_extra = {
            "example": {
                "label_space" : [
                    {
                        "id" : 1, "text" : "NER-TYPE-1", "user_provided" : True
                    },
                    {
                        "id" : 2, "text" : "NER-TYPE-2", "user_provided" : True
                    },
                    {
                        "id" : 3, "text" : "NER-TYPE-3", "user_provided" : True
                    },
                    {
                        "id" : 4, "text" : "relation-1", "user_provided" : False
                    },
                    {
                        "id" : 5, "text" : "relation-2", "user_provided" : False
                    }
                ],
                "annotated" : [
                    {
                        "text" : "This is some random text for our relation extraction example, that we're going to insert SUBJ and OBJ into.",
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER-TYPE-1",
                                "start_offset" : 13, 
                                "end_offset" : 19,
                                "user_provided" : True
                            },
                            {
                                "id" : 2,
                                "label_text" : "NER-TYPE-2",
                                "start_offset" : 53, 
                                "end_offset" : 57,
                                "user_provided" : True
                            },
                            {
                                "id" : 3,
                                "label_text" : "NER-TYPE-3",
                                "start_offset" : 62, 
                                "end_offset" : 69,
                                "user_provided" : True
                            },
                            {
                                "id" : 4,
                                "sbj_start_offset" : 13,
                                "sbj_end_offset" : 19,
                                "obj_start_offset" : 53,
                                "obj_end_offset" : 57,
                                "label_text" : "relation-1",
                                "user_provided" : False
                            },
                            {
                                "id" : 5,
                                "sbj_start_offset" : 62,
                                "sbj_end_offset" : 69,
                                "obj_start_offset" : 53,
                                "obj_end_offset" : 57,
                                "label_text" : "relation-2",
                                "user_provided" : False
                            }
                        ],
                        "explanations" : [
                            {
                                "annotation_id" : 4,
                                "text" : "This is the first explanation for annotation 4"
                            },
                            {
                                "annotation_id" : 4,
                                "text" : "This is the second explanation for annotation 4"
                            },
                            {
                                "annotation_id" : 4,
                                "text" : "This is the third explanation for annotation 4"
                            },
                            {
                                "annotation_id" : 5,
                                "text" : "Annotation 5 has only one explanation"
                            }
                        ]
                    },
                    {
                        "text" : "This document has no relations, but we can still use it for training!",
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER-TYPE-1",
                                "start_offset" : 5, 
                                "end_offset" : 13,
                                "user_provided" : True
                            },
                            {
                                "id" : 2,
                                "label_text" : "NER-TYPE-1",
                                "start_offset" : 21, 
                                "end_offset" : 30,
                                "user_provided" : True
                            },
                            {
                                "id" : 3,
                                "label_text" : "NER-TYPE-3",
                                "start_offset" : 60, 
                                "end_offset" : 68,
                                "user_provided" : True
                            }
                        ],
                        "explanations" : [
                        ]
                    }
                ],
                "unlabeled" : [
                    {
                        "text" : "Some unlabeled text", 
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER-TYPE-1",
                                "start_offset" : 0, 
                                "end_offset" : 4,
                                "user_provided" : True
                            },
                            {
                                "id" : 2,
                                "label_text" : "NER-TYPE-2",
                                "start_offset" : 5, 
                                "end_offset" : 14,
                                "user_provided" : True
                            }
                        ]
                    },
                    {
                        "text" : "This won't be considered unlabeled text for relation extraction, too few entities. Though it will be considered in dataset_size.",
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER-TYPE-1",
                                "start_offset" : 0, 
                                "end_offset" : 4,
                                "user_provided" : True
                            }
                        ]
                    },
                    {"text" : "This also won't be considered unlabeled text for relation extraction. However, for sentiment analysis, all text is considered valid."}
                ]
            }
        }
class LeanLifePayload(BaseModel):
    lean_life_data : LeanLifeData
    params : LeanLifeParams

class ExplanationTriple(BaseModel):
    text : str
    explanation : str
    label : str

class ExplanationTrainingPayload(BaseModel):
    params : TrainingApiParams
    label_space : Dict[str, int]
    explanation_triples : Optional[List[ExplanationTriple]]
    unlabeled_text : Optional[List[str]]
    ner_label_space : Optional[List[str]]

    class Config:
        schema_extra = {
            "example": {
                "params" : {
                    "stage" : "both",
                    "experiment_name" : "test_experiment_1",
                    "dataset_name" : "test_dataset",
                    "dataset_size" : 3,
                    "task" : "re",
                    "pre_train_build_data" : True,
                    "build_data" : True,
                    "relation_ner_types" : {
                        "label-1" : ("PERSON", "PERSON"),
                        "label-2" : ("PERSON", "DATE"),
                        "label-3" : ("ORGANIZATION", "ORGANIZATION"),
                        "label-4" : ("PERSON", "PERSON")
                    },
                    "none_label_key" : "no_relation"
                },
                "label_space" : {
                    "label-1" : 0,
                    "label-2" : 1,
                    "label-3" : 2, 
                    "label-4" : 3,
                    "no_relation" : 4
                },
                "explanation_triples" : [
                    {
                        "text" : "SUBJ-PERSON 's daughter OBJ-PERSON said Tuesday that her uncle was `` doing very well '' in his lengthy recovery , and was following very closely a gender equality bill under debate.",
                        "explanation" : "The phrase \"'s daughter\" links SUBJ and OBJ and there are no more than three words between SUBJ and OBJ",
                        "label" : "label-1"
                    },
                    {
                        "text" : "SUBJ-PERSON was born OBJ-DATE , in Nashville , Tenn , and graduated with honors from the University of Alabama.",
                        "explanation" : "SUBJ and OBJ sandwich the phrase \"was born\" and there are no more than three words between SUBJ and OBJ",
                        "label" : "label-2"
                    },
                    {
                        "text" : "Under the agreement , AT&T will begin offering SUBJ-ORGANIZATION as part of its OBJ-ORGANIZATION service after Jan. 31 , when AT&T 's current agreement with Dish Network expires.",
                        "explanation" : "There are no more than five words between SUBJ and OBJ and \"as part of its\" appears between SUBJ and OBJ",
                        "label" : "label-3"
                    },
                    {
                        "text" : "SUBJ-PERSON , who died of OBJ-CAUSE_OF_DEATH Monday at the age of 78 , was a complicated person , and any attempt to sum up her life and work will necessarily turn into a string of contradictions.",
                        "explanation" : "Between SUBJ and OBJ the phrase \"who died of\" occurs and there are no more than five words between SUBJ and OBJ",
                        "label" : "label-2"
                    },
                    {
                        "text" : "The style and concept is inspired by three generations of women in their family , with the name `` Der\u00e9on '' paying tribute to SUBJ-PERSON 's grandmother , OBJ-PERSON.",
                        "explanation" : "The phrase \"'s grandmother\" occurs between SUBJ and OBJ and there are no more than four words between SUBJ and OBJ",
                        "label" : "label-4"
                    }
                ],
                "unlabeled_text" : [
                    "At the same time, Chief Financial Officer SUBJ-PERSON will become OBJ-TITLE, succeeding Stephen Green who is leaving to take a government job.",
                    "U.S. District Court Judge OBJ-PERSON in mid-February issued an injunction against Wikileaks after the Zurich-based Bank SUBJ-PERSON accused the site of posting sensitive account information stolen by a disgruntled former employee.",
                    "OBJ-CITY 2009-07-07 11:07:32 UTC French media earlier reported that SUBJ-PERSON , ranked 119 , was found dead by his girlfriend in the stairwell of his Paris apartment."
                ],
                "ner_label_space" : ["PERSON", "CITY", "TITLE", "ORGANIZATION", "CAUSE_OF_DEATH", "DATE"]
            }
        }

class MatchedDataOutput(BaseModel):
    matched_tuples = List[Tuple[str, str]]
    matched_indices = List[Tuple[str, str]]

    class Config:
        schema_extra = {
            "example": {
                "matched_tuples" : [
                    ("At the same time, Chief Financial Officer SUBJ-PERSON will become OBJ-TITLE, succeeding Stephen Green who is leaving to take a government job.", "label-1"),
                    ("Two competing battery makers -- Compact Power Inc. of Troy , Michigan , which is working with parent LG Chem of Korea , and Frankfurt , Germany-based Continental Automotive Systems , which is working with OBJ-ORGANIZATION and SUBJ-ORGANIZATION of Watertown , Massachusetts -- fell 10 weeks behind on delivering the power packs.", "label-2"),
                ],
                "matched_indices" : [
                    (0, 0),
                    (3, 2)
                ]
            }
        }
class StrictMatchPayload(BaseModel):
    explanation_triples : List[ExplanationTriple]
    unlabeled_text : List[str]
    task : str

    class Config:
        schema_extra = {
            "example": {
                "explanation_triples" : [
                    {
                        "text" : "SUBJ-PERSON 's daughter OBJ-PERSON said Tuesday that her uncle was `` doing very well '' in his lengthy recovery , and was following very closely a gender equality bill under debate.",
                        "explanation" : "The phrase \"'s daughter\" links SUBJ and OBJ and there are no more than three words between SUBJ and OBJ",
                        "label" : "label-1"
                    },
                    {
                        "text" : "SUBJ-PERSON was born OBJ-DATE , in Nashville , Tenn , and graduated with honors from the University of Alabama.",
                        "explanation" : "SUBJ and OBJ sandwich the phrase \"was born\" and there are no more than three words between SUBJ and OBJ",
                        "label" : "label-2"
                    },
                    {
                        "text" : "Under the agreement , AT&T will begin offering SUBJ-ORGANIZATION as part of its OBJ-ORGANIZATION service after Jan. 31 , when AT&T 's current agreement with Dish Network expires.",
                        "explanation" : "There are no more than five words between SUBJ and OBJ and \"as part of its\" appears between SUBJ and OBJ",
                        "label" : "label-3"
                    },
                    {
                        "text" : "SUBJ-PERSON , who died of OBJ-CAUSE_OF_DEATH Monday at the age of 78 , was a complicated person , and any attempt to sum up her life and work will necessarily turn into a string of contradictions.",
                        "explanation" : "Between SUBJ and OBJ the phrase \"who died of\" occurs and there are no more than five words between SUBJ and OBJ",
                        "label" : "label-2"
                    },
                    {
                        "text" : "The style and concept is inspired by three generations of women in their family , with the name `` Der\u00e9on '' paying tribute to SUBJ-PERSON 's grandmother , OBJ-PERSON.",
                        "explanation" : "The phrase \"'s grandmother\" occurs between SUBJ and OBJ and there are no more than four words between SUBJ and OBJ",
                        "label" : "label-4"
                    }
                ],
                "unlabeled_text" : [
                    "At the same time, Chief Financial Officer SUBJ-PERSON will become OBJ-TITLE, succeeding Stephen Green who is leaving to take a government job.",
                    "U.S. District Court Judge OBJ-PERSON in mid-February issued an injunction against Wikileaks after the Zurich-based Bank SUBJ-PERSON accused the site of posting sensitive account information stolen by a disgruntled former employee.",
                    "OBJ-CITY 2009-07-07 11:07:32 UTC French media earlier reported that SUBJ-PERSON , ranked 119 , was found dead by his girlfriend in the stairwell of his Paris apartment.",
                    "Two competing battery makers -- Compact Power Inc. of Troy , Michigan , which is working with parent LG Chem of Korea , and Frankfurt , Germany-based Continental Automotive Systems , which is working with OBJ-ORGANIZATION and SUBJ-ORGANIZATION of Watertown , Massachusetts -- fell 10 weeks behind on delivering the power packs.",
                    "Berkshire shareholders voted Wednesday to split the company 's Class B shares 50-for-1 in a move tied to OBJ-ORGANIZATION 's $ 26.3 billion acquisition of SUBJ-ORGANIZATION.",
                    "The chiefs of more than 60 top companies support the Conservatives ' position and on Thursday the executive chairman of OBJ-NATIONALITY retail giant Marks & Spencer , SUBJ-PERSON , attacked the prime minister for dismissing their concerns."
                ],
                "task" : "re"
            }
        }

class NextEvalDataOutput(BaseModel):
    avg_loss : float
    avg_eval_ent_f1_score : float
    avg_eval_val_f1_score : float
    no_relation_thresholds : Tuple[float, float]

    class Config:
        schema_extra = {
            "example": {
                "avg_loss" : 0.03012,
                "avg_eval_ent_f1_score" : 43.32233,
                "avg_eval_val_f1_score" : 40.5654323,
                "no_relation_thresholds" : (2.912, 0.931)
            }
        }

class EvalNextClfPayload(BaseModel):
    params : EvalApiParams
    label_space : Dict[str, int]
    eval_data : List[Tuple[str, str]]


    class Config:
        schema_extra = {
            "example": {
                "params" : {
                    "experiment_name" : "test_experiment_1",
                    "dataset_name" : "test_dataset",
                    "train_dataset_size" : 3,
                    "task" : "re",
                    "none_label_key" : "no_relation"
                },
                "label_space" : {
                    "label-1" : 0,
                    "label-2" : 1,
                    "label-3" : 2, 
                    "label-4" : 3,
                    "no_relation" : 4
                },
                "eval_data" : [
                    ("He has served as a policy aide to the late U.S. Senator Alan Cranston , as National Issues Director for the 2004 presidential campaign of Congressman Dennis Kucinich , as a co-founder of SUBJ-ORGANIZATION and as a member of the OBJ-ORGANIZATION at the RAND Corporation think tank before all that.", "no_relation"),
                    ("SUBJ-PERSON , who had since flown around 50 sorties , was promoted posthumously from lieutenant to OBJ-TITLE , the military spokeswoman said , adding that the date of his funeral will be announced later.", "label-2")
                ]
            }
        }