import json
import constants as const

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
    subj_start_offset = annotation["start_offset_1"]
    subj_end_offset = annotation["end_offset_1"]
    obj_start_offset = annotation["start_offset_2"]
    obj_end_offset = annotation["end_offset_2"]
    label = annotation["label_text"]

    subj_ner_key = _create_ner_key(subj_start_offset, subj_end_offset)
    obj_ner_key = _create_ner_key(obj_start_offset, obj_end_offset)
    subj_ner_type = ner_mentions[subj_ner_key]
    obj_ner_type = ner_mentions[obj_ner_key]

    subj_first = subj_start_offset < obj_start_offset
    if subj_first:
        relation_text = text[:subj_start_offset]
        relation_text += "SUBJ-{}".format(subj_ner_type)
        relation_text += text[subj_end_offset+1:obj_start_offset]
        relation_text += "OBJ-{}".format(obj_ner_type)
        relation_text += text[obj_end_offset+1:]
        relation_text = relation_text.strip()
    else:
        relation_text = text[:obj_start_offset]
        relation_text += "OBJ-{}".format(obj_ner_type)
        relation_text += text[obj_end_offset+1:subj_start_offset]
        relation_text += "SUBJ-{}".format(subj_ner_type)
        relation_text += text[subj_end_offset+1:]
        relation_text = relation_text.strip()
    
    return relation_text, label

def _generate_no_label_pairs(ner_mentions, text):
    """
        When an annotater indicates that a document has no relations between its entities, we create training
        points that indicate that all possible pairs of entities should have the label "no relation".

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
                start_offset_1, end_offset_1 = key_1.split("-")
                start_offset_2, end_offset_2 = key_2.split("-")
                start_offset_1 = int(start_offset_1)
                end_offset_1 = int(end_offset_1)
                start_offset_2 = int(start_offset_2)
                end_offset_2 = int(end_offset_2)
                annotation = {
                    "start_offset_1" : start_offset_1,
                    "end_offset_1" : end_offset_1,
                    "start_offset_2" : start_offset_2,
                    "end_offset_2" : end_offset_2,
                    "label_text" : ""
                }
                relation_text, label = _create_re_pair(annotation, text, ner_mentions)
                relation_texts.append(relation_text)
                labels.append(label)
    return relation_texts, labels

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
    text = annotated_doc["text"]
    explanations = annotated_doc["explanations"]
    annotations = annotated_doc["annotations"]
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
            temp_explanation_triples[annotation_id].append((relation_text, label, explanation_text))
        else:
            temp_explanation_triples[annotation_id] = [(relation_text, label, explanation_text)]

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
    text = annotated_doc["text"]
    explanations = annotated_doc["explanations"]
    annotations = annotated_doc["annotations"]
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
            temp_explanation_triples[annotation_id].append((text, label, explanation_text))
        else:
            temp_explanation_triples[annotation_id] = [(text, label, explanation_text)]

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
                                                        dictionary where, key-annotation_id
                                                        value-array of of (sentence, label, explanation) triples
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
    
    return list(sentence_label_pairs.values()), explanation_triples

def _read_lean_life_dataset(file_obj):
    """
        Read in a LEAN_LIFE Dataset file and convert it into the needed datastructures for training

        Arguments:
            file_obj (File) : Python File Object
        
        Returns:
            training_pairs, explanation_triples, label_space, unlabeled_docs : arr, dict, arr, arr
    """
    data = json.load(file_obj)
    project_type = data["project_type"]
    label_space = data["label_space"]
    label_space = [label["text"] for label in label_space if not label["user_provided"]]
    unlabeled_docs = data["unlabeled"]
    unlabeled_docs = [doc["text"] for doc in unlabeled_docs]
    annotated_docs = data["annotated"]
    training_pairs, explanation_triples = _process_annotations(annotated_docs, project_type)
    
    return training_pairs, explanation_triples, label_space, unlabeled_docs

def kickstart_training(file_obj, lean_life=True):
    """
        Given a post request indicating a training job should be kicked off, organize data into the needed
        datastructures for training. Then call the `start_training` function

        Arguments:
            file_obj  (File) : Python File Object
            lean_life (bool) : boolean indicating whether file is a lean_life file or a user provided file
    """
    if lean_life:
        training_pairs, explanation_triples, label_space, unlabeled_docs = read_lean_life_dataset(file_obj)
    else:
        data = json.load(file_obj)
        training_pairs = data["training_pairs"]
        explanation_triples = data["explanation_triples"]
        label_space = data["label_space"]
        unlabeled_docs = data["unlabeled_text"]
    
    # start_training(training_pairs, explanation_triples, label_space, unlabeled_docs)
    return
