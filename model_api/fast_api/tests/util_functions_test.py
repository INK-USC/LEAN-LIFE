import json
import sys
sys.path.append("../")
import fast_api_constants as const
import json_schema as schema
import fast_api_util_functions as util_f

with open("re_test_dataset.json") as f:
    re_data = json.load(f)

re_data = schema.LeanLifeData.parse_obj(re_data)

with open("sa_test_dataset.json") as f:
    sa_data = json.load(f)

sa_data = schema.LeanLifeData.parse_obj(sa_data)

def test_create_ner_key():
    start_offset = 10
    end_offset = 20
    key = util_f._create_ner_key(start_offset, end_offset)
    assert key == "10-20"

def test_create_re_pair():
    text = "this is some random text, that we're going to insert subj and objects into."
    ner_mentions = {
        "13-19" : "RAND",
        "53-57" : "TEMP",
        "62-69" : "NER_TYPE"
    }

    annotation = {
        "sbj_start_offset" : 13,
        "sbj_end_offset" : 19,
        "obj_start_offset" : 53,
        "obj_end_offset" : 57,
        "label_text" : "relation-1",
    }
    
    actual_relation_text = "this is some SUBJ-RAND text, that we're going to insert OBJ-TEMP and objects into."
    actual_label = "relation-1"
    
    relation_text, label = util_f._create_re_pair(annotation, text, ner_mentions)

    assert actual_relation_text == relation_text
    assert actual_label == label

    annotation = {
        "sbj_start_offset" : 62,
        "sbj_end_offset" : 69,
        "obj_start_offset" : 53,
        "obj_end_offset" : 57,
        "label_text" : "relation-2",
    }

    actual_relation_text = "this is some random text, that we're going to insert OBJ-TEMP and SUBJ-NER_TYPE into."
    actual_label = "relation-2"

    relation_text, label = util_f._create_re_pair(annotation, text, ner_mentions)
    
    assert actual_relation_text == relation_text
    assert actual_label == label

def test_generate_no_label_pairs():
    text = "this is some random text, that we're going to insert subj and objects into."
    ner_mentions = {
        "13-19" : "RAND",
        "53-57" : "TEMP",
        "62-69" : "NER_TYPE"
    }

    actual_relation_text = "this is some SUBJ-RAND text, that we're going to insert OBJ-TEMP and objects into."

    relation_texts, labels = util_f._generate_no_label_pairs(ner_mentions, text)
    
    assert 6 == len(relation_texts)
    assert actual_relation_text == relation_texts[0]
    assert 6 == len(labels)
    assert "" == labels[0]

def test_process_re_annotated_doc_with_annotations():
    annotated_docs = re_data.annotated
    annotated_doc = annotated_docs[0]
    fake_id = -1

    actual_sentence_label_pairs = {
        4: ("this is some SUBJ-RAND text, that we're going to insert OBJ-TEMP and objects into.", 'relation-1'), 
        5: ("this is some random text, that we're going to insert OBJ-TEMP and SUBJ-NER_TYPE into.", 'relation-2')
    }

    actual_explanation_triples = {
        4 : { 
                'text' : "this is some SUBJ-RAND text, that we're going to insert OBJ-TEMP and objects into.", 
                'label' : 'relation-1', 
                'explanation' : [
                    'This is the first explanation for annotation 4', 
                    'This is the second explanation for annotation 4',
                    'This is the third explanation for annotation 4'
                ]
            }, 
        5: {
                'text' : "this is some random text, that we're going to insert OBJ-TEMP and SUBJ-NER_TYPE into.",
                'label' : 'relation-2',
                'explanation': ['Annotation 5 has only one explanation']
            }
    }

    actual_fake_id = -1
    
    temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id = util_f._process_re_annotated_doc(annotated_doc, fake_id)

    assert actual_sentence_label_pairs == temp_sentence_label_pairs
    assert actual_explanation_triples == temp_explanation_triples
    assert actual_fake_id == temp_fake_id
    
def test_process_re_annotated_doc_no_annotations():
    annotated_docs = re_data.annotated
    annotated_doc = annotated_docs[1]
    fake_id = -1

    actual_sentence_label_pairs = {
        -1: ('this SUBJ-TEMP has no OBJ-TEMP, but we can still use it for training!', ''),
        -2: ('this SUBJ-TEMP has no relations, but we can still use it for OBJ-NER_TYPE!', ''),
        -3: ('this OBJ-TEMP has no SUBJ-TEMP, but we can still use it for training!', ''),
        -4: ('this document has no SUBJ-TEMP, but we can still use it for OBJ-NER_TYPE!', ''),
        -5: ('this OBJ-TEMP has no relations, but we can still use it for SUBJ-NER_TYPE!', ''),
        -6: ('this document has no OBJ-TEMP, but we can still use it for SUBJ-NER_TYPE!', '')
    }

    actual_explanation_triples = {}

    actual_fake_id = -7

    temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id = util_f._process_re_annotated_doc(annotated_doc, fake_id)

    assert actual_sentence_label_pairs == temp_sentence_label_pairs
    assert actual_explanation_triples == temp_explanation_triples
    assert actual_fake_id == temp_fake_id

def test_process_re_unlabeled_doc():
    unlabeled_docs = re_data.unlabeled
    valid_doc = unlabeled_docs[0]

    actual_unlabeled_text = [
        'SUBJ-TEMP1 OBJ-TEMP2 text',
        'OBJ-TEMP1 SUBJ-TEMP2 text'
    ]

    assert actual_unlabeled_text == util_f._procress_re_unlabeled_doc(valid_doc)

    invalid_doc = unlabeled_docs[1]
    actual_unlabeled_text = []

    assert actual_unlabeled_text == util_f._procress_re_unlabeled_doc(invalid_doc)

def test_process_sa_annotated_doc_with_explanation():
    annotated_docs = sa_data.annotated
    annotated_doc = annotated_docs[0]
    fake_id = -1

    actual_sentence_label_pairs = {
        1: ("this is some random text that we're going to say has a positive label.", 'pos') 
    }

    actual_explanation_triples = {
        1 : {
                'text' : "this is some random text that we're going to say has a positive label.",
                'label' : 'pos',
                'explanation' : [
                    'This is the first explanation for annotation 1',
                    'This is the second explanation for annotation 1'
                ]
            }
    }

    actual_fake_id = -1
    
    temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id = util_f._process_sa_annotated_doc(annotated_doc, fake_id)
    
    assert actual_sentence_label_pairs == temp_sentence_label_pairs
    assert actual_explanation_triples == temp_explanation_triples
    assert actual_fake_id == temp_fake_id

def test_process_sa_annotated_doc_no_explanation():
    annotated_docs = sa_data.annotated
    annotated_doc = annotated_docs[1]
    fake_id = -1

    actual_sentence_label_pairs = {
        2: ("this is some random text that we're going to say has a negative label.", 'neg') 
    }

    actual_explanation_triples = {
    }

    actual_fake_id = -1
    
    temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id = util_f._process_sa_annotated_doc(annotated_doc, fake_id)
    
    assert actual_sentence_label_pairs == temp_sentence_label_pairs
    assert actual_explanation_triples == temp_explanation_triples
    assert actual_fake_id == temp_fake_id

def test_process_sa_annotated_doc_no_annotation():
    annotated_docs = sa_data.annotated
    annotated_doc = annotated_docs[2]
    fake_id = -1

    actual_sentence_label_pairs = {
        -1: ("this is some random text that we're going to say has no label.", '') 
    }

    actual_explanation_triples = {
    }

    actual_fake_id = -2
    
    temp_sentence_label_pairs, temp_explanation_triples, temp_fake_id = util_f._process_sa_annotated_doc(annotated_doc, fake_id)
    
    assert actual_sentence_label_pairs == temp_sentence_label_pairs
    assert actual_explanation_triples == temp_explanation_triples
    assert actual_fake_id == temp_fake_id

def test_process_annotations_sa():
    annotated_docs = sa_data.annotated
    
    actual_sentence_label_pairs = [
        ("this is some random text that we're going to say has a positive label.", 'pos'),
        ("this is some random text that we're going to say has a negative label.", 'neg'),
        ("this is some random text that we're going to say has no label.", '')
    ]

    actual_explanation_triples = [
        {
            'text' : "this is some random text that we're going to say has a positive label.",
            'label' : 'pos',
            'explanation' : [
                'This is the first explanation for annotation 1',
                'This is the second explanation for annotation 1'
            ]
        }
    ]

    sentence_label_pairs, explanation_triples = util_f._process_annotations(annotated_docs, const.LEAN_LIFE_SA_PROJECT)
    assert set(actual_sentence_label_pairs) == set(sentence_label_pairs)
    assert actual_explanation_triples == explanation_triples

def test_process_annotations_re():
    annotated_docs = re_data.annotated
    
    actual_sentence_label_pairs = [
        ("this is some SUBJ-RAND text, that we're going to insert OBJ-TEMP and objects into.", 'relation-1'),
        ("this is some random text, that we're going to insert OBJ-TEMP and SUBJ-NER_TYPE into.", 'relation-2'),
        ('this SUBJ-TEMP has no OBJ-TEMP, but we can still use it for training!', ''),
        ('this SUBJ-TEMP has no relations, but we can still use it for OBJ-NER_TYPE!', ''),
        ('this OBJ-TEMP has no SUBJ-TEMP, but we can still use it for training!', ''),
        ('this document has no SUBJ-TEMP, but we can still use it for OBJ-NER_TYPE!', ''),
        ('this OBJ-TEMP has no relations, but we can still use it for SUBJ-NER_TYPE!', ''),
        ('this document has no OBJ-TEMP, but we can still use it for SUBJ-NER_TYPE!', '')
    ]

    actual_explanation_triples = [
        { 
            'text' : "this is some SUBJ-RAND text, that we're going to insert OBJ-TEMP and objects into.", 
            'label' : 'relation-1', 
            'explanation' : [
                'This is the first explanation for annotation 4', 
                'This is the second explanation for annotation 4',
                'This is the third explanation for annotation 4'
            ]
        }, 
        {
            'text' : "this is some random text, that we're going to insert OBJ-TEMP and SUBJ-NER_TYPE into.",
            'label' : 'relation-2',
            'explanation': ['Annotation 5 has only one explanation']
        }
    ]

    sentence_label_pairs, explanation_triples = util_f._process_annotations(annotated_docs, const.LEAN_LIFE_RE_PROJECT)
    assert set(actual_sentence_label_pairs) == set(sentence_label_pairs)
    assert actual_explanation_triples == explanation_triples

def test_read_lean_life_dataset_sa():
    label_space, unlabeled_docs, explanation_triples, ner_label_space, training_pairs = util_f._read_lean_life_dataset(sa_data, const.LEAN_LIFE_SA_PROJECT)

    actual_training_pairs = [
        ("this is some random text that we're going to say has a positive label.", 'pos'),
        ("this is some random text that we're going to say has a negative label.", 'neg'),
        ("this is some random text that we're going to say has no label.", '')
    ]

    actual_explanation_triples = [
        {
            'text' : "this is some random text that we're going to say has a positive label.",
            'label' : 'pos',
            'explanation' : [
                'This is the first explanation for annotation 1',
                'This is the second explanation for annotation 1'
            ]
        }
    ]

    actual_label_space = ["pos", "neg"]
    actual_unlabeled_docs = [
        'some unlabeled text',
        'for the machine to train off',
        'normally there is lots of unlabeled text',
        'however, this is just a test',
        'so there is limited text'
    ]

    assert set(actual_training_pairs) == set(training_pairs)
    assert actual_explanation_triples == explanation_triples
    assert set(actual_label_space) == set(label_space)
    assert set(actual_unlabeled_docs) == set(unlabeled_docs)
    assert ner_label_space == []

def test_read_lean_life_dataset_re():
    label_space, unlabeled_docs, explanation_triples, ner_label_space, training_pairs = util_f._read_lean_life_dataset(re_data, const.LEAN_LIFE_RE_PROJECT)
    
    actual_training_pairs = [
        ("this is some SUBJ-RAND text, that we're going to insert OBJ-TEMP and objects into.", 'relation-1'),
        ("this is some random text, that we're going to insert OBJ-TEMP and SUBJ-NER_TYPE into.", 'relation-2'),
        ('this SUBJ-TEMP has no OBJ-TEMP, but we can still use it for training!', ''),
        ('this SUBJ-TEMP has no relations, but we can still use it for OBJ-NER_TYPE!', ''),
        ('this OBJ-TEMP has no SUBJ-TEMP, but we can still use it for training!', ''),
        ('this document has no SUBJ-TEMP, but we can still use it for OBJ-NER_TYPE!', ''),
        ('this OBJ-TEMP has no relations, but we can still use it for SUBJ-NER_TYPE!', ''),
        ('this document has no OBJ-TEMP, but we can still use it for SUBJ-NER_TYPE!', '')
    ]

    actual_explanation_triples = [
        { 
            'text' : "this is some SUBJ-RAND text, that we're going to insert OBJ-TEMP and objects into.", 
            'label' : 'relation-1', 
            'explanation' : [
                'This is the first explanation for annotation 4', 
                'This is the second explanation for annotation 4',
                'This is the third explanation for annotation 4'
            ]
        }, 
        {
            'text' : "this is some random text, that we're going to insert OBJ-TEMP and SUBJ-NER_TYPE into.",
            'label' : 'relation-2',
            'explanation': ['Annotation 5 has only one explanation']
        }
    ]

    actual_label_space = ["relation-1", "relation-2"]
    actual_ner_label_space = ['RAND', 'TEMP', 'NER_TYPE']
    actual_unlabeled_docs = [
        'SUBJ-TEMP1 OBJ-TEMP2 text',
        'OBJ-TEMP1 SUBJ-TEMP2 text'
    ]

    assert set(actual_training_pairs) == set(training_pairs)
    assert actual_explanation_triples == explanation_triples
    assert set(actual_label_space) == set(label_space)
    assert set(actual_ner_label_space) == set(ner_label_space)
    assert set(actual_unlabeled_docs) == set(unlabeled_docs)