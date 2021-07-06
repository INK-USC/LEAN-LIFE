import sys
sys.path.append("../")
import json
import pdb
import pickle
from CCG.parser import CCGParserTrainer, TrainedCCGParser
from CCG.CCG_utils import prepare_token_for_rule_addition, _find_quoted_phrases

ec_ccg_trainer = CCGParserTrainer(task="ec", explanation_file="data/ec_test_data.json",
                                  unlabeled_data_file="data/carer_test_data.json")

re_ccg_trainer = CCGParserTrainer(task="re", explanation_file="data/tacred_test_explanation_data.json",
                                  unlabeled_data_file="data/tacred_test_unlabeled_data.json")

def test_trainer_load_data_ec():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    loaded_data = ec_ccg_trainer.parser.loaded_data
    assert len(loaded_data) == 16
    for datapoint in loaded_data:
        assert "DataPoint" in str(type(datapoint))

def test_trainer_load_data_re():
    explanation_file = re_ccg_trainer.params["explanation_file"]
    re_ccg_trainer.load_data(explanation_file)
    loaded_data = re_ccg_trainer.parser.loaded_data
    assert len(loaded_data) == 10
    for datapoint in loaded_data:
        assert "DataPoint" in str(type(datapoint))

def test_trainer_prepare_unlabeled_data_ec():
    unlabeled_data_file = ec_ccg_trainer.params["unlabeled_data_file"]
    ec_ccg_trainer.prepare_unlabeled_data(unlabeled_data_file, cache=False)
    unlabeled_data = ec_ccg_trainer.unlabeled_data
    assert len(unlabeled_data) == 1000
    for phrase in unlabeled_data:
        assert "Phrase" in str(type(phrase))

def test_trainer_prepare_unlabeled_data_re():
    unlabeled_data_file = re_ccg_trainer.params["unlabeled_data_file"]
    re_ccg_trainer.prepare_unlabeled_data(unlabeled_data_file, cache=False)
    unlabeled_data = re_ccg_trainer.unlabeled_data
    assert len(unlabeled_data) == 1002
    for phrase in unlabeled_data:
        assert "Phrase" in str(type(phrase))

def test_parser_create_and_set_grammar_ec():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser_grammar = parser.grammar
    with open(explanation_file) as f:
        explanation_data = json.load(f)
    
    for dic in explanation_data:
        token = "\"" + dic["word"].lower() + "\""
        prepped_token = prepare_token_for_rule_addition(dic["word"].lower())
        assert prepped_token in parser_grammar

def test_parser_create_and_set_grammar_re():
    explanation_file = re_ccg_trainer.params["explanation_file"]
    re_ccg_trainer.load_data(explanation_file)
    parser = re_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser_grammar = parser.grammar
    with open(explanation_file) as f:
        explanation_data = json.load(f)
        explanation_data = [entry["explanation"] for entry in explanation_data]
    
    words = []
    for explanation in explanation_data:
        quoted_words = _find_quoted_phrases(explanation)
        for word in quoted_words:
            words.append(word)
    
    for word in words:
        prepped_token = prepare_token_for_rule_addition(word.lower())
        assert prepped_token in parser_grammar

def test_parser_tokenize_explanations_ec():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    loaded_data = parser.loaded_data
    expected_ec_tokenizations = [
        [['$The', '$Sentence', '$Contains', '$The', '$Word', '"angry"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"increasingSPACEanger"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"beenSPACEsoSPACEexcited"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"whenSPACEtheSPACEgtaSPACEonlineSPACEbikerSPACEdlcSPACEcomesSPACEout"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"bigot"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"howSPACEshit"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"terrorSPACEthreat"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"panic"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"smiling"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"cheerySPACEsmile"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"soSPACEsad"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"mySPACEdepression"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"bitSPACEsurprised"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"soSPACEsudden"']],
         [['$The', '$Sentence', '$Contains', '$The', '$Word', '"honesty"']],
         [['$The', '$Word', '"dependentSPACEonSPACEanotherSPACEperson"', '$In', '$The', '$Sentence']]
    ]

    assert len(loaded_data) == 16
    for i, datapoint in enumerate(loaded_data):
        assert datapoint.tokenized_explanations == expected_ec_tokenizations[i]

def test_parser_tokenize_explanations_re():
    explanation_file = re_ccg_trainer.params["explanation_file"]
    re_ccg_trainer.load_data(explanation_file)
    parser = re_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    loaded_data = parser.loaded_data
    expected_re_tokenizations = [
        [['$The', '$Word', '"PUNCT5sSPACEdaughter"', '$Link', '$ArgX', '$And', '$ArgY', '$And', '$There', '$Is', '$AtMost', '"3"', '$Word', '$Between', '$ArgX', '$And', '$ArgY']],
        [['$ArgX', '$And', '$ArgY', '$SandWich', '$The', '$Word', '"wasSPACEborn"', '$And', '$There', '$Is', '$AtMost', '"3"', '$Word', '$Between', '$ArgX', '$And', '$ArgY']],
        [['$There', '$Is', '$AtMost', '"5"', '$Word', '$Between', '$ArgX', '$And', '$ArgY', '$And', '"asSPACEpartSPACEofSPACEits"', '$Is', '$Between', '$ArgX', '$And', '$ArgY']],
        [['$Between', '$ArgX', '$And', '$ArgY', '$The', '$Word', '"whoSPACEdiedSPACEof"', '$Is', '$And', '$There', '$Is', '$AtMost', '"5"', '$Word', '$Between', '$ArgX', '$And', '$ArgY']],
        [['$Between', '$ArgX', '$And', '$ArgY', '$The', '$Word', '"isSPACEfrom"', '$Is', '$And', '$There', '$Is', '$AtMost', '"4"', '$Word', '$Between', '$ArgX', '$And', '$ArgY']],
        [['$The', '$Word', '"PUNCT5sSPACEgrandmother"', '$Is', '$Between', '$ArgX', '$And', '$ArgY', '$And', '$There', '$Is', '$AtMost', '"4"', '$Word', '$Between', '$ArgX', '$And', '$ArgY']],
        [['$The', '$Word', '"isSPACEbasedSPACEin"', '$Is', '$Between', '$ArgX', '$And', '$ArgY', '$And', '$There', '$Is', '$AtMost', '"5"', '$Word', '$Between', '$ArgX', '$And', '$ArgY']],
        [['$The', '$Word', '"thenSPACEknownSPACEas"', '$Is', '$Between', '$ArgX', '$And', '$ArgY', '$And', '$There', '$Is', '$AtMost', '"6"', '$Word', '$Between', '$ArgX', '$And', '$ArgY']],
        [['$ArgX', '$And', '$ArgY', '$SandWich', '$The', '$Word', '"PUNCT5sSPACEmother"', '$And', '$There', '$Is', '$AtMost', '"3"', '$Word', '$Between', '$ArgX', '$And', '$ArgY']],
        [['$There', '$Is', '$AtMost', '"4"', '$Word', '$Between', '$ArgX', '$And', '$ArgY', '$And', '$ArgX', '$And', '$ArgY', '$SandWich', '$The', '$Word', '"secretlySPACEmarried"']]
    ]

    assert len(loaded_data) == 10
    for i, datapoint in enumerate(loaded_data):
        assert datapoint.tokenized_explanations == expected_re_tokenizations[i]


def test_parser_build_labeling_rules_ec():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    parser.build_labeling_rules()
    loaded_data = parser.loaded_data

    semantic_counts = [
        {('.root', ('@In1', 'Sentence', ('@Word', 'angry'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'increasing anger'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'been so excited'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'when the gta online biker dlc comes out'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'bigot'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'how shit'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'terror threat'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'panic'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'smiling'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'cheery smile'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'so sad'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'my depression'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'bit surprised'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'so sudden'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'honesty'))): 1},
        {('.root', ('@In1', 'Sentence', ('@Word', 'dependent on another person'))): 1}
    ]

    assert len(loaded_data) == 16

    for i, datapoint in enumerate(loaded_data):
        assert datapoint.semantic_counts == semantic_counts[i]
        keys = list(datapoint.semantic_counts.keys())
        for key in keys:
            assert key in datapoint.labeling_functions

def test_parser_build_labeling_rules_re():
    explanation_file = re_ccg_trainer.params["explanation_file"]
    re_ccg_trainer.load_data(explanation_file)
    parser = re_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    parser.build_labeling_rules()
    loaded_data = parser.loaded_data

    semantic_counts = [
        {('.root', ('@And', ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '3', 'tokens'))), ('@Is', ('@Word', "'s daughter"), ('@between', ('@And', 'ArgY', 'ArgX'))))): 1},
        {('.root', ('@And', ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '3', 'tokens'))), ('@Is', ('@Word', 'was born'), ('@between', ('@And', 'ArgY', 'ArgX'))))): 1},
        {('.root', ('@And', ('@Is', 'as part of its', ('@between', ('@And', 'ArgY', 'ArgX'))), ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '5', 'tokens'))))): 1},
        {('.root', ('@And', ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '5', 'tokens'))), ('@Is', ('@Word', 'who died of'), ('@between', ('@And', 'ArgY', 'ArgX'))))): 1},
        {('.root', ('@And', ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '4', 'tokens'))), ('@Is', ('@Word', 'is from'), ('@between', ('@And', 'ArgY', 'ArgX'))))): 1},
        {('.root', ('@And', ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '4', 'tokens'))), ('@Is', ('@Word', "'s grandmother"), ('@between', ('@And', 'ArgY', 'ArgX'))))): 1},
        {('.root', ('@And', ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '5', 'tokens'))), ('@Is', ('@Word', 'is based in'), ('@between', ('@And', 'ArgY', 'ArgX'))))): 1},
        {('.root', ('@And', ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '6', 'tokens'))), ('@Is', ('@Word', 'then known as'), ('@between', ('@And', 'ArgY', 'ArgX'))))): 1},
        {('.root', ('@And', ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '3', 'tokens'))), ('@Is', ('@Word', "'s mother"), ('@between', ('@And', 'ArgY', 'ArgX'))))): 1},
        {('.root', ('@And', ('@Is', ('@Word', 'secretly married'), ('@between', ('@And', 'ArgY', 'ArgX'))), ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '4', 'tokens'))))): 1}
    ]

    assert len(loaded_data) == 10
    labeling_function_semantics = []
    for i, datapoint in enumerate(loaded_data):
        assert datapoint.semantic_counts == semantic_counts[i]
        keys = list(datapoint.semantic_counts.keys())
        for key in keys:
            assert key in datapoint.labeling_functions

def test_filter_matrix_ec():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    parser.build_labeling_rules()
    
    # If the below file needs to be changed, this is how it was generated
    # ec_ccg_trainer.prepare_unlabeled_data(ec_ccg_trainer.params["unlabeled_data_file"])
    # with open("data/carer_test_data_phrase.p", "wb") as f:
    #     pickle.dump(ec_ccg_trainer.unlabeled_data, f)
    # unlabeled_data = ec_ccg_trainer.unlabeled_data

    with open("data/carer_test_data_phrase.p", "rb") as f:
        unlabeled_data = pickle.load(f)

    parser.low_end_filter_count = 0
    parser.matrix_filter(unlabeled_data, task="ec")
    # filter count set to zero, so certain explanations don't fire at all on this small sample
    # hence hash filter filters some out
    label_counts = {"anger" : 2, "fear" : 1, "joy" : 1, "sadness" : 1, "surprise" : 1, "trust" : 1}
    final_labeling_functions = parser.labeling_functions
    semantic_reps = parser.semantic_reps
    filtered_raw_explanations = parser.filtered_raw_explanations

    actual_filtered_raw_explanations = {
        ('.root', ('@In1', 'Sentence', ('@Word', 'angry'))): "The tweet contains the phrase 'angry'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'increasing anger'))): "The tweet contains the phrase 'increasing anger'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'panic'))): "The tweet contains the phrase 'panic'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'smiling'))): "The tweet contains the phrase 'smiling'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'my depression'))): "The tweet contains the phrase 'my depression'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'bit surprised'))): "The tweet contains the phrase 'bit surprised'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'honesty'))): "The tweet contains the phrase 'honesty'"
    }

    final_label_counts = {}
    for key in final_labeling_functions:
        assert "function" in str(type(key))
        emotion = final_labeling_functions[key]
        if emotion in final_label_counts:
            final_label_counts[emotion] += 1
        else:
            final_label_counts[emotion] = 1
    
    for key in final_label_counts:
        assert final_label_counts[key] == label_counts[key]
    
    for key in actual_filtered_raw_explanations:
        assert key in semantic_reps
        assert semantic_reps[key] in final_labeling_functions
        assert key in filtered_raw_explanations
        assert filtered_raw_explanations[key] == actual_filtered_raw_explanations[key]
    
    assert parser.ner_types == None
    
    parser.low_end_filter_count = 1
    parser.labeling_functions = None
    parser.matrix_filter(unlabeled_data, task="ec")
    # filter count set to 1, so certain explanations don't fire at all on this small sample
    # hence count filter filters some out
    # anger has one less count, as hash keeps the first misfiring one
    label_counts = {"anger" : 1, "fear" : 1, "joy" : 1, "sadness" : 1, "surprise" : 1, "trust" : 1}
    final_labeling_functions = parser.labeling_functions
    semantic_reps = parser.semantic_reps
    filtered_raw_explanations = parser.filtered_raw_explanations

    actual_filtered_raw_explanations = {
        ('.root', ('@In1', 'Sentence', ('@Word', 'angry'))): "The tweet contains the phrase 'angry'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'panic'))): "The tweet contains the phrase 'panic'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'smiling'))): "The tweet contains the phrase 'smiling'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'my depression'))): "The tweet contains the phrase 'my depression'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'bit surprised'))): "The tweet contains the phrase 'bit surprised'",
        ('.root', ('@In1', 'Sentence', ('@Word', 'honesty'))): "The tweet contains the phrase 'honesty'"
    }

    final_label_counts = {}
    for key in final_labeling_functions:
        assert "function" in str(type(key))
        emotion = final_labeling_functions[key]
        if emotion in final_label_counts:
            final_label_counts[emotion] += 1
        else:
            final_label_counts[emotion] = 1
    
    for key in final_label_counts:
        assert final_label_counts[key] == label_counts[key]
    
    for key in actual_filtered_raw_explanations:
        assert key in semantic_reps
        assert semantic_reps[key] in final_labeling_functions
        assert key in filtered_raw_explanations
        assert filtered_raw_explanations[key] == actual_filtered_raw_explanations[key]
    
    assert parser.ner_types == None

def test_filter_matrix_re():
    explanation_file = re_ccg_trainer.params["explanation_file"]
    re_ccg_trainer.load_data(explanation_file)
    parser = re_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    parser.build_labeling_rules()
    
    # If the below file needs to be changed, this is how it was generated
    # re_ccg_trainer.prepare_unlabeled_data(re_ccg_trainer.params["unlabeled_data_file"])
    # with open("data/tacred_test_unlabeled_data_phrase.p", "wb") as f:
    #     pickle.dump(re_ccg_trainer.unlabeled_data, f)
    # unlabeled_data = re_ccg_trainer.unlabeled_data
    
    with open("data/tacred_test_unlabeled_data_phrase.p", "rb") as f:
        unlabeled_data = pickle.load(f)

    parser.low_end_filter_count = 0
    parser.matrix_filter(unlabeled_data)
    # filter count set to zero, so certain explanations don't fire at all on this small sample
    # hence hash filter filters some out
    label_counts = {'per:children': 1, 'per:spouse': 1}
    final_labeling_functions = parser.labeling_functions
    semantic_reps = parser.semantic_reps
    filtered_raw_explanations = parser.filtered_raw_explanations

    actual_filtered_raw_explanations = {
        ('.root', ('@And', ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '3', 'tokens'))), ('@Is', ('@Word', "'s daughter"), ('@between', ('@And', 'ArgY', 'ArgX'))))): 'The phrase "\'s daughter" links SUBJ and OBJ and there are no more than three words between SUBJ and OBJ',
        ('.root', ('@And', ('@Is', ('@Word', 'secretly married'), ('@between', ('@And', 'ArgY', 'ArgX'))), ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '4', 'tokens'))))): 'There are no more than four words between SUBJ and OBJ and SUBJ and OBJ sandwich the phrase "secretly married"'
    }
    
    final_label_counts = {}
    for key in final_labeling_functions:
        assert "function" in str(type(key))
        relation = final_labeling_functions[key]
        if relation in final_label_counts:
            final_label_counts[relation] += 1
        else:
            final_label_counts[relation] = 1
        
    for key in label_counts:
        assert final_label_counts[key] == label_counts[key]
    
    for key in actual_filtered_raw_explanations:
        assert key in semantic_reps
        assert semantic_reps[key] in final_labeling_functions
        assert key in filtered_raw_explanations
        assert filtered_raw_explanations[key] == actual_filtered_raw_explanations[key]
    
    assert parser.ner_types != None
    
    parser.low_end_filter_count = 1
    parser.labeling_functions = None
    parser.matrix_filter(unlabeled_data)
    # # filter count set to 1, so certain explanations don't fire at all on this small sample
    # # hence count filter filters some out
    label_counts = {'per:spouse': 1}
    final_labeling_functions = parser.labeling_functions
    semantic_reps = parser.semantic_reps
    filtered_raw_explanations = parser.filtered_raw_explanations

    actual_filtered_raw_explanations = {
        ('.root', ('@And', ('@Is', ('@Word', 'secretly married'), ('@between', ('@And', 'ArgY', 'ArgX'))), ('@Is', 'There', ('@AtMost', ('@between', ('@And', 'ArgY', 'ArgX')), ('@Num', '4', 'tokens'))))): 'There are no more than four words between SUBJ and OBJ and SUBJ and OBJ sandwich the phrase "secretly married"'
    }

    final_label_counts = {}
    for key in final_labeling_functions:
        assert "function" in str(type(key))
        relation = final_labeling_functions[key]
        if relation in final_label_counts:
            final_label_counts[relation] += 1
        else:
            final_label_counts[relation] = 1
        
    for key in label_counts:
        assert final_label_counts[key] == label_counts[key]
    
    for key in actual_filtered_raw_explanations:
        assert key in semantic_reps
        assert semantic_reps[key] in final_labeling_functions
        assert key in filtered_raw_explanations
        assert filtered_raw_explanations[key] == actual_filtered_raw_explanations[key]
    
    assert parser.ner_types != None

def test_parser_build_soft_labeling_rules_ec():
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    parser.build_labeling_rules()
    with open("data/carer_test_data_phrase.p", "rb") as f:
        unlabeled_data = pickle.load(f)
    parser.low_end_filter_count = 0
    parser.matrix_filter(unlabeled_data, task="ec")
    parser.build_soft_labeling_functions()
    
    actual_filtered_explanations = [
        "The tweet contains the phrase 'angry'", 
        "The tweet contains the phrase 'increasing anger'",
        "The tweet contains the phrase 'panic'",
        "The tweet contains the phrase 'smiling'",
        "The tweet contains the phrase 'my depression'",
        "The tweet contains the phrase 'bit surprised'",
        "The tweet contains the phrase 'honesty'"
    ]

    actual_function_labels = ['anger', 'anger', 'fear', 'joy', 'sadness', 'surprise', 'trust']

    outputted_labels = []
    for i, pair in enumerate(parser.soft_labeling_functions):
        function, relation = pair
        assert "function" in str(type(function))
        outputted_labels.append(relation)
    
    assert set(actual_function_labels) == set(outputted_labels)
    assert set(actual_filtered_explanations) == set(list(parser.filtered_raw_explanations.values()))

def test_parser_build_soft_labeling_rules_re():
    explanation_file = re_ccg_trainer.params["explanation_file"]
    re_ccg_trainer.load_data(explanation_file)
    parser = re_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser.tokenize_explanations()
    parser.build_labeling_rules()
    with open("data/tacred_test_unlabeled_data_phrase.p", "rb") as f:
        unlabeled_data = pickle.load(f)
    parser.low_end_filter_count = 0
    parser.matrix_filter(unlabeled_data)
    parser.build_soft_labeling_functions()

    actual_filtered_explanations = [
        'The phrase "\'s daughter" links SUBJ and OBJ and there are no more than three words between SUBJ and OBJ',
        'There are no more than four words between SUBJ and OBJ and SUBJ and OBJ sandwich the phrase "secretly married"'
    ]

    actual_function_labels = ['per:children', 'per:spouse']
    outputted_labels = []
    for i, pair in enumerate(parser.soft_labeling_functions):
        function, relation = pair
        assert "function" in str(type(function))
        outputted_labels.append(relation)
    
    assert set(actual_function_labels) == set(outputted_labels)
    assert set(actual_filtered_explanations) == set(list(parser.filtered_raw_explanations.values()))