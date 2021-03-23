import json
import sys
import pathlib
PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
import fast_api_constants as const
import requests

def _create_ner_key(start_offset, end_offset):
    """
        Create a key out of start and end offsets. Useful for identifying which NER mentions are involved
        in a relation.

        Arguments:
            start_offset (int) : start index of the ner mention (inclusive)
            end_offset   (int) : end index of the ner mention (inclusive)
        
        Returns:
            str : string representation of an NER mention's position in a string.
    """
    return "-".join([str(start_offset), str(end_offset)])    

def _create_re_pair(annotation, text, ner_mentions):
    """
        Parses an annotation and returns the needed text and label for training
        Text is adjusted to reflect the provided annotation. NER mentions that make up the provided
        relation are replaced with SUBJ-NER_TYPE and OBJ-NER_TYPE to indicate to the machine the 
        tokens that should be used for final classification.

        Arguments:
            annotation   (dict) : dictionary representing an re annnotation
            text          (str) : the text of the document that the annotation is made upon
            ner_mentions (dict) : key-output of _create_ner_key, value-NER type
        
        Returns:
            relation_text, label : text where NERs that form the relation are indicated/normalized,
                                   label is a string that completes the relation (subj, label, obj)
    """
    subj_start_offset = annotation["sbj_start_offset"]
    subj_end_offset = annotation["sbj_end_offset"]
    obj_start_offset = annotation["obj_start_offset"]
    obj_end_offset = annotation["obj_end_offset"]
    label = annotation["label_text"]

    subj_ner_key = _create_ner_key(subj_start_offset, subj_end_offset)
    obj_ner_key = _create_ner_key(obj_start_offset, obj_end_offset)
    subj_ner_type = ner_mentions[subj_ner_key]
    obj_ner_type = ner_mentions[obj_ner_key]

    subj_first = subj_start_offset < obj_start_offset
    if subj_first:
        relation_text = text[:subj_start_offset]
        relation_text += "SUBJ-{}".format(subj_ner_type)
        relation_text += text[subj_end_offset:obj_start_offset]
        relation_text += "OBJ-{}".format(obj_ner_type)
        relation_text += text[obj_end_offset:]
        relation_text = relation_text.strip()
    else:
        relation_text = text[:obj_start_offset]
        relation_text += "OBJ-{}".format(obj_ner_type)
        relation_text += text[obj_end_offset:subj_start_offset]
        relation_text += "SUBJ-{}".format(subj_ner_type)
        relation_text += text[subj_end_offset:]
        relation_text = relation_text.strip()
    
    return relation_text, label

def _generate_no_label_pairs(ner_mentions, text):
    """
        When an annotater indicates that a document has no relations between its entities, we create training
        points that indicate that all possible pairs of entities should have the label "no relation".

        If len(ner_mentions) < 2, empty arrays are sent back

        Arguments:
            ner_mentions (dict) : all the NER mentions in the text, key - entity position, value - entity type
            text          (str) : the text of the document
        
        Returns:
            relation_texts, labels: array of text where the NERs that form the relation are 
                                    indicated/normalized. labels is an array where each element is the empty
                                    string.

    """
    mention_keys = list(ner_mentions.keys())
    relation_texts = []
    labels = []
    for i in range(len(mention_keys)):
        for j in range(len(mention_keys)):
            if i != j:
                key_1 = mention_keys[i]
                key_2 = mention_keys[j]
                sbj_start_offset, sbj_end_offset = key_1.split("-")
                obj_start_offset, obj_end_offset = key_2.split("-")
                sbj_start_offset = int(sbj_start_offset)
                sbj_end_offset = int(sbj_end_offset)
                obj_start_offset = int(obj_start_offset)
                obj_end_offset = int(obj_end_offset)
                annotation = {
                    "sbj_start_offset" : sbj_start_offset,
                    "sbj_end_offset" : sbj_end_offset,
                    "obj_start_offset" : obj_start_offset,
                    "obj_end_offset" : obj_end_offset,
                    "label_text" : ""
                }
                relation_text, label = _create_re_pair(annotation, text, ner_mentions)
                relation_texts.append(relation_text)
                labels.append(label)
    return relation_texts, labels

def _procress_re_unlabeled_doc(unlabeled_doc):
    """
        For each unlabeled document, we prepare the text to indicate what subj and obj to try and classify
        as having a relation. We do it for every pair of named_entities provided in the unlabeled document.

        Arguments:
            unlabeled_doc (dict) : dictionary representation of an unlabeled document
        
        Returns:
            arr : array of text where the NERs that form the relation are indicated/normalized
    """
    text = unlabeled_doc.text
    annotations = unlabeled_doc.annotations
    ner_mentions = {}
    for annotation in annotations:
        if annotation["user_provided"]:
            key = _create_ner_key(annotation["start_offset"], annotation["end_offset"])
            label = annotation["label_text"]
            ner_mentions[key] = label
    
    relation_texts, _ = _generate_no_label_pairs(ner_mentions, text)

    return relation_texts

def _process_re_annotated_doc(annotated_doc, fake_id):
    """
        1. Extract all the labeled data from an annotated document and create training pairs
        2. Generate labeled data points for documents with no relations
        3. Extract all the explanations from an annotated document and groups them by the 
           annotation they are associated with. Instead of chaining explanations together by
           and, we evaluate each clause seperatley and then do the needful and operations later.

        Arguments:
            annotated_doc (dict) : dictionary representation of an annotated document
            fake_id        (int) : starting id to use for annotations indicating no label
        
        Returns:
            temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id: Training data points extracted,
                                                                               Explanations extracted,
                                                                               Updated fake id
    """
    temp_sentence_label_pairs = {}
    temp_explanation_triples = {}
    temp_fake_id = fake_id
    ner_mentions = {}
    re_data = []
    text = annotated_doc.text
    explanations = annotated_doc.explanations
    annotations = annotated_doc.annotations
    for annotation in annotations:
        if annotation["user_provided"]:
            key = _create_ner_key(annotation["start_offset"], annotation["end_offset"])
            label = annotation["label_text"]
            ner_mentions[key] = label
        else:
            re_data.append(annotation)
    
    for annotation in re_data:
        relation_text, label = _create_re_pair(annotation, text, ner_mentions)
        temp_sentence_label_pairs[annotation["id"]] = (relation_text, label)
    
    if len(re_data) == 0:
        relation_texts, labels = _generate_no_label_pairs(ner_mentions, text)
        for i, text in enumerate(relation_texts):
            temp_sentence_label_pairs[temp_fake_id] = (text, labels[i])
            temp_fake_id += -1
    
    for explanation in explanations:
        annotation_id = explanation["annotation_id"]
        explanation_text = explanation["text"]
        relation_text = temp_sentence_label_pairs[annotation_id][0]
        label = temp_sentence_label_pairs[annotation_id][1]
        if annotation_id in temp_explanation_triples:
            temp_explanation_triples[annotation_id]["explanation"].append(explanation_text)
        else:
            temp_explanation_triples[annotation_id] = {
                "text" : relation_text,
                "label" : label,
                "explanation" : [explanation_text]
            }

    return temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id

def _process_sa_annotated_doc(annotated_doc, fake_id):
    """
        1. Extract all the labeled data from an annotated document and create training pairs, else indicate
           no_label (empty string)
        2. Extract all the explanations from an annotated document and groups them by the 
           annotation they are associated with. Instead of chaining explanations together by
           and, we evaluate each clause seperatley and then do the needful and operations later.

        Arguments:
            annotated_doc (dict) : dictionary representation of an annotated document
            fake_id        (int) : starting id to use for annotations indicating no label
        
        Returns:
            temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id: Training data points extracted,
                                                                               Explanations extracted,
                                                                               Updated fake id
    """
    temp_sentence_label_pairs = {}
    temp_explanation_triples = {}
    temp_fake_id = fake_id
    text = annotated_doc.text
    explanations = annotated_doc.explanations
    annotations = annotated_doc.annotations
    if len(annotations):
        for annotation in annotations:
            temp_sentence_label_pairs[annotation["id"]] = (text, annotation["label_text"])
    else:
        temp_sentence_label_pairs[temp_fake_id] = (text, "")
        temp_fake_id += -1

    for explanation in explanations:
        annotation_id = explanation["annotation_id"]
        explanation_text = explanation["text"]
        label = temp_sentence_label_pairs[annotation_id][1]
        if annotation_id in temp_explanation_triples:
            temp_explanation_triples[annotation_id]["explanation"].append(explanation_text)
        else:
            temp_explanation_triples[annotation_id] = {
                "text" : text,
                "label" : label,
                "explanation" : [explanation_text]
            }

    return temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id

def _process_annotations(annotated_docs, project_type):
    """
        Given a list of annotated documents and a project type, this function creates the needed datasets
        for training.

        Arguments:
            annotated_docs (arr) : list of annotated docs
            project_type   (int) : integer representing what type of annotations to expect
        
        Returns:
            sentence_label_pairs, explanation_triples : array of (sentence, label) pairs,
                                                        array of (dict("text" : str, "label" : str, "explanation" : List(str)))
    """
    sentence_label_pairs = {}
    explanation_triples = {}
    fake_id = -1
    processing_func = None
    
    if project_type == const.LEAN_LIFE_SA_PROJECT:
        processing_func = _process_sa_annotated_doc
    elif project_type == const.LEAN_LIFE_RE_PROJECT:
        processing_func = _process_re_annotated_doc 
    
    for doc in annotated_docs:
        if processing_func:
            processed_doc = processing_func(doc, fake_id)
        else:
            processed_doc = ({}, {}, fake_id)
        temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id = processed_doc
        fake_id = temp_fake_id
        for key in temp_sentence_label_pairs:
            sentence_label_pairs[key] = temp_sentence_label_pairs[key]
        for key in temp_explanation_triples:
            explanation_triples[key] = temp_explanation_triples[key]
    
    return list(sentence_label_pairs.values()), list(explanation_triples.values())

def _read_lean_life_dataset(json_data, project_type):
    """
        Read in a LEAN_LIFE Dataset file and convert it into the needed datastructures for training

        Arguments:
            json_data (LeanLifeData) : json_schema.LeanLifeData
            project_type       (str) : string indicating project type

        Returns:
            label_space, unlabeled_docs, explanation_triples, ner_label_space, training_pairs,  : dict, arr, arr, arr, arr
    """
    label_space = json_data.label_space
    label_space = [label.text for label in label_space if not label.user_provided]
    label_space = {label : i for i, label in enumerate(label_space)}
    ner_label_space = [label.text for label in label_space if label.user_provided and len(label.text) > 0]
    unlabeled_docs = json_data.unlabeled
    if project_type != const.LEAN_LIFE_RE_PROJECT:
        unlabeled_docs = [doc.text for doc in unlabeled_docs]
    else:
        annotated_unlabeled_docs = [doc for doc in unlabeled_docs if hasattr(doc, "annotations")]
        unlabeled_docs = [text for doc in annotated_unlabeled_docs for text in _procress_re_unlabeled_doc(doc)]
    annotated_docs = json_data.annotated
    if len(annoted_docs):
        training_pairs, explanation_triples = _process_annotations(annotated_docs, project_type)
    else:
        training_pairs, explanation_triples = [], []
    
    return label_space, unlabeled_docs, explanation_triples, ner_label_space, training_pairs

def prepare_next_data(json_data, project_type="", lean_life=True):
    """
        Given a post request indicating a training job should be kicked off, organize data into the needed
        datastructures for training. It doesn't fully complete the pre-processing, allowing for flexibility
        in the future. For example, currently we "and" all explanations associated with an annotation (internal_main.py),
        that policy will probably be changed. So these are more useful intermediary data-structures. We currently
        create training_pairs as well, but these are not being used.

        Arguments:
            json_data (LeanLifeData|ExplanationTrainingPayload) : json_schema.LeanLifeData or json_schema.ExplanationTrainingPayload
            project_type                                  (str) : string indicating project type for lean_life data
            lean_life                                    (bool) : boolean indicating whether file is a lean_life file or a user provided payload
    """
    if lean_life:
        label_space, unlabeled_docs, explanation_triples, ner_label_space, training_pairs = _read_lean_life_dataset(json_data, project_type)
    else:
        # training_pairs = json_data.training_pairs
        label_space = json_data.label_space
        if hasattr(json_data, "unlabeled_text"):
            unlabeled_docs = json_data.unlabeled_text
        else:
            unlabeled_docs = []
        if hasattr(json_data, "explanation_triples"):
            explanation_triples = json_data.explanation_triples
        else:
            explanation_triples = []
        if hasattr(json_data, "ner_label_space"):
            ner_label_space = json_data.ner_label_space
        else:
            ner_label_space = []
    
    return label_space, unlabeled_docs, explanation_triples, ner_label_space

def update_model_training(experiment_name, cur_epoch, total_epochs, time_spent, best_train_loss, stage):
    """
        Sends a POST request back to LEAN-LIFE Django API, updating the API with the latest model training
        status. Django API takes this information and writes to a local file, so that when the front-end
        requests an update, the Django API can read from this file and update accordingly.

        Arguments:
            experiment_name      (str) : name of model being trained
            cur_epoch            (int) : current epoch for training
            total_epochs         (int) : total epochs needed for training
            time_spent           (int) : how much time has passed in seconds
            best_train_loss    (float) : best training loss so far
            stage                (str) : message to indicate stage of pipeline, and what stage the time
                                         estimate is referring to
        
        Returns:
            (int) : status code from Django API after receiving request
    """
    if cur_epoch > 0 :
        finished_pct = cur_epoch / total_epochs
        approximate_total_time = time_spent / finished_pct
        time_left = int(approximate_total_time - time_spent)
    else:
       time_left = -1 

    update_repr = {
        experiment_name : {
            "cur_epoch" : cur_epoch,
            "total_epochs" : total_epochs,
            "time_spent" : time_spent,
            "time_left" : time_left,
            "best_train_loss" : best_train_loss,
            "stage" : stage
        }
    }

    end_point = const.LEAN_LIFE_URL + "update/training_status/"
    response = requests.post(end_point, json=update_repr)

    return response.status_code

def send_model_metadata(experiment_name, save_path, best_train_loss=None, file_size=None):
    """
        Sends a POST request back to LEAN-LIFE Django API, updating the API with the fact that the model has
        been saved, and where the model can be found when the model needs to be downloaded.

        Arguments:
            experiment_name   (str) : name of model being trained
            save_path         (str) : where the model has been saved
            best_train_loss (float) : best training loss so far
            file_size         (int) : size of model params file in bytes
        
        Returns:
            (int) : status code from Django API after receiving request
    """
    if best_train_loss and file_size:
        is_trained = True
        metadata_repr = {
            experiment_name : {
                "is_trained" : is_trained,
                "save_path" : save_path,
                "best_train_loss" : best_train_loss,
                "file_size" : file_size
            }
        }
    else:
        is_trained = False
        metadata_repr = {
            experiment_name : {
                "is_trained" : is_trained,
                "save_path" : save_path
            }
        }

    end_point = const.LEAN_LIFE_URL + "update/models_metadata/"
    response = requests.post(end_point, json=metadata_repr)

    return response.status_code