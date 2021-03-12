import spacy
import sys
import pickle
import json
sys.path.append("../")
from CCG import utils
from CCG.parser import CCGParserTrainer, TrainedCCGParser

nlp = spacy.load("en_core_web_sm")

def test_find_quoted_phrases():
    text = 'First type of "query"'

    assert ['"query"'] == utils._find_quoted_phrases(text)

    text = 'Another "type" of "query"'

    assert ['"type"', '"query"'] == utils._find_quoted_phrases(text)

    text = 'Ideally all explanations will only use "double quote\'s", so we can avoid issues with "\'"'

    assert ['"double quote\'s"', '"\'"'] == utils._find_quoted_phrases(text)

    text = "An explanation can use 'single quotes'"

    assert ["'single quotes'"] == utils._find_quoted_phrases(text)

    text = "However, there can be some problems with 'apostrophes like, 's'"

    assert ["'apostrophes like, '"] == utils._find_quoted_phrases(text)

    text = "We can even handle ''double single quotes too''"

    assert ["'double single quotes too'"] == utils._find_quoted_phrases(text)

    text = "Though do \"not\" mix 'quotes'"

    assert ['"not"'] == utils._find_quoted_phrases(text)

    text = "Finally we also handle `backticks as quotes`"

    assert ["`backticks as quotes`"] == utils._find_quoted_phrases(text)

    text = "No quotes here though, so should be empty"

    assert [] == utils._find_quoted_phrases(text)
    
def test_clean_text():
    text = "Rahul"
    assert utils.clean_text(text) == "rahul"
    assert utils.clean_text(text, lower=False, exclusive=False) == "Rahul"

    text = "Rahul, what if I want to keep punctuation!?!?!?."
    assert utils.clean_text(text) == "rahul, what if i want to keep punctuation."
    assert utils.clean_text(text, collapse_punc=False, exclusive=False) == "rahul, what if i want to keep punctuation!?!?!?."
    
    text = "RAHUL,,,,, why did you break it!?!?!?!."
    assert utils.clean_text(text) == "rahul, why did you break it."
    assert utils.clean_text(text, commas=False) == "rahul,,,,, why did you break it."

    text = "Rahul, maybe its a good idea to drink more water & less coffee."
    assert utils.clean_text(text) == "rahul, maybe its a good idea to drink more water and less coffee."
    assert utils.clean_text(text, switch_amp=False, exclusive=False) == "rahul, maybe its a good idea to drink more water & less coffee."

    text = "RahUL, look it's a tweet #hashtags @username could be USEFUL, but we also lose tokens like $%^"
    assert utils.clean_text(text) ==  "rahul, look it's a tweet hashtags username could be useful, but we also lose tokens like"
    assert utils.clean_text(text, exclusive=False) == "rahul, look it's a tweet #hashtags @username could be useful, but we also lose tokens like $%^"
    
    text = "Rahul    sometimes there is like    weird   whitespace!"
    assert utils.clean_text(text) == "rahul sometimes there is like weird whitespace."
    assert utils.clean_text(text, whitespace=False) == "rahul    sometimes there is like    weird   whitespace."

def test_segment_explanation():
    text = 'First type of "query"'

    assert ["First type of ", "*\"query\""] == utils.segment_explanation(text)

    text = 'Another "type" of "query"'

    assert ["Another ", "*\"type\"", " of ", "*\"query\""] == utils.segment_explanation(text)

    text = 'No queries here'

    assert ["No queries here"] == utils.segment_explanation(text)
    
def test_clean_explanation():
    explanation = "The tweet contains the phrase 'cheery smile'"
    assert utils.clean_explanation(explanation) == "the tweet contains the phrase \"cheery smile\""

    explanation = "The tweet contains the phrase 'Cheery smile'"
    assert utils.clean_explanation(explanation) == "the tweet contains the phrase \"cheery smile\""

    explanation = "The tweet contains some funky characters like '$$$!!!@@#()#)$*$' & BETWEEN SUBJ AND OBJ THERE are four Words"
    assert utils.clean_explanation(explanation) == "the tweet contains some funky characters like \"$$$!!!@@#()#)$*$\" and between subj and obj there are four words"

    
def test_convert_chunk_to_terminal():
    terminal_chunks = ["not", "n\"t", "n't", "number", "/"]
    terminals = [["$Not"], ["$Not"], ["$Not"], ['$Count', '$NumberNER'], ['$Separator']]
    for i, chunk in enumerate(terminal_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == terminals[i]
    
    quote_chunks = ["\"this is a quote chunk\"", "\"and this is another one\""]
    for i, chunk in enumerate(quote_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == [chunk]
    
    digit_chunks = ["4", "3", "10", "100"]
    digit_quotes = [["\"4\""], ["\"3\""], ["\"10\""], ["\"100\""]]
    for i, chunk in enumerate(digit_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == digit_quotes[i]
    
    num_chunks = ["one", "two", "ten"]
    num_quotes = [["\"1\""], ["\"2\""], ["\"10\""]]
    for i, chunk in enumerate(num_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == num_quotes[i]
    
    not_recognized_chunks = ["glue", "blue", "shoe", "boo boo", "gaboo"]
    for i, chunk in enumerate(not_recognized_chunks):
        assert utils.convert_chunk_to_terminal(chunk) == None    

def test_chunk_explanation():
    explanation = "The words ', is a' appear right before OBJ and the word 'citizen' is right after OBJ"
    expected_chunking = ['the', 'words', '", is a"', 'appear', 'right', 'before', 'obj', 'and', 'the', 'word', '"citizen"', 'is', 'right', 'after', 'obj']
    cleaned_explanation = utils.clean_explanation(explanation)
    chunked_explanation = utils.chunk_explanation(cleaned_explanation, nlp)
    assert chunked_explanation == expected_chunking

    explanation = "The words 'year old' are to the right of OBJ and 'is' is right before OBJ"         
    expected_chunking = ['the', 'words', '"year old"', "are", "$Right", "obj", "and", '"is"', "is", "right", "before", "obj"]
    cleaned_explanation = utils.clean_explanation(explanation)
    chunked_explanation = utils.chunk_explanation(cleaned_explanation, nlp)
    assert chunked_explanation == expected_chunking

    explanation = "The tweet contains the phrase 'when the GTA Online Biker DLC comes out'"
    expected_chunking = ['the', 'tweet', 'contains', 'the', 'phrase', '"when the gta online biker dlc comes out"']
    cleaned_explanation = utils.clean_explanation(explanation)
    chunked_explanation = utils.chunk_explanation(cleaned_explanation, nlp)
    assert chunked_explanation == expected_chunking

def test_prepare_token_for_rule_addition():
    quote_token = "\"this could be important!!:)\""
    expected_prep = "\"thisSPACEcouldSPACEbeSPACEimportantPUNCT0PUNCT0PUNCT14PUNCT7\""
    prepped_token = utils.prepare_token_for_rule_addition(quote_token)
    assert prepped_token == expected_prep
    
    reverse_token = utils.prepare_token_for_rule_addition(expected_prep, reverse=True)
    assert reverse_token == quote_token

def test_generate_phrase():
    sentence = "His wife, OBJ-PERSON, often accompanied him on SUBJ-PERSON SUBJ-PERSON expeditions, as she did in 1947, when she became the first woman to climb Mount McKinley"
    phrase_tokens = ['His', 'wife', ',', 'OBJ', ',', 'often', 'accompanied', 'him', 'on', 'SUBJ', 'expeditions', ',', 'as', 'she', 'did', 'in', '1947', ',', 'when', 'she', 'became', 'the', 'first', 'woman', 'to', 'climb', 'Mount', 'McKinley']
    # the last two NER labels are wrong, but are the output of spaCy's NER tagger.
    phrase_ners = ['', '', '', 'PERSON', '', '', '', '', '', 'PERSON', '', '', '', '', '', '', 'DATE', '', '', '', '', '', 'ORDINAL', '', '', '', 'PERSON', 'PERSON']
    phrase_subj_posi = 9
    phrase_obj_posi = 3
    phrase = utils.generate_phrase(sentence, nlp)
    assert len(phrase.tokens) == len(phrase.ners)
    assert phrase.tokens == phrase_tokens
    assert phrase.ners == phrase_ners
    assert phrase.subj_posi == phrase_subj_posi
    assert phrase.obj_posi == phrase_obj_posi

    sentence = "SUBJ-PERSON's mother OBJ-PERSON was a singer in the dance group Soul II Soul, which had hits in the 1980s and 1990s."
    phrase_tokens = ['SUBJ', "'s", 'mother', 'OBJ', 'was', 'a', 'singer', 'in', 'the', 'dance', 'group', 'Soul', 'II', 'Soul', ',', 'which', 'had', 'hits', 'in', 'the', '1980s', 'and', '1990s', '.']
    phrase_ners = ['PERSON', '', '', 'PERSON', '', '', '', '', '', '', '', 'PRODUCT', 'PRODUCT', 'PRODUCT', '', '', '', '', '', '', 'DATE', '', 'DATE', '']
    phrase_subj_posi = 0
    phrase_obj_posi = 3
    phrase = utils.generate_phrase(sentence, nlp)
    assert len(phrase.tokens) == len(phrase.ners)
    assert phrase.tokens == phrase_tokens
    assert phrase.ners == phrase_ners
    assert phrase.subj_posi == phrase_subj_posi
    assert phrase.obj_posi == phrase_obj_posi

    sentence = "GAMEDAY VS BUFORD TODAY AT 5:30 AT HOME ! ! ! NEVER BEEN SO EXCITED #revenge"
    phrase_tokens = ['GAMEDAY', 'VS', 'BUFORD', 'TODAY', 'AT', '5:30', 'AT', 'HOME', '!', '!', '!', 'NEVER', 'BEEN', 'SO', 'EXCITED', '#', 'revenge']
    phrase_ners = ['ORG', '', '', '', '', 'TIME', '', '', '', '', '', '', '', '', '', '', '']
    phrase_subj_posi = len(phrase_tokens)
    phrase_obj_posi = len(phrase_tokens)
    phrase = utils.generate_phrase(sentence, nlp)
    assert len(phrase.tokens) == len(phrase.ners)
    assert phrase.tokens == phrase_tokens
    assert phrase.ners == phrase_ners
    assert phrase.subj_posi == phrase_subj_posi
    assert phrase.obj_posi == phrase_obj_posi

def test_parse_tokens_re():
    re_ccg_trainer = CCGParserTrainer(task="re", explanation_file="data/tacred_test_explanation_data.json",
                                      unlabeled_data_file="data/tacred_test_unlabeled_data.json")
    explanation_file = re_ccg_trainer.params["explanation_file"]
    re_ccg_trainer.load_data(explanation_file)
    parser = re_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser_grammar = parser.grammar
    
    re_tokens = [
        ['$The', '$Word', '"PUNCT5sSPACEdaughter"', '$Link', '$ArgX', '$And', '$ArgY', '$And', '$There', '$Is', '$AtMost', '"3"', '$Word', '$Between', '$ArgX', '$And', '$ArgY'],
        ['$ArgX', '$And', '$ArgY', '$SandWich', '$The', '$Word', '"wasSPACEborn"', '$And', '$There', '$Is', '$AtMost', '"3"', '$Word', '$Between', '$ArgX', '$And', '$ArgY'],
        ['$There', '$Is', '$AtMost', '"5"', '$Word', '$Between', '$ArgX', '$And', '$ArgY', '$And', '"asSPACEpartSPACEofSPACEits"', '$Is', '$Between', '$ArgX', '$And', '$ArgY'],
        ['$Between', '$ArgX', '$And', '$ArgY', '$The', '$Word', '"whoSPACEdiedSPACEof"', '$Is', '$And', '$There', '$Is', '$AtMost', '"5"', '$Word', '$Between', '$ArgX', '$And', '$ArgY'],
        ['$Between', '$ArgX', '$And', '$ArgY', '$The', '$Word', '"isSPACEfrom"', '$Is', '$And', '$There', '$Is', '$AtMost', '"4"', '$Word', '$Between', '$ArgX', '$And', '$ArgY'],
        ['$The', '$Word', '"PUNCT5sSPACEgrandmother"', '$Is', '$Between', '$ArgX', '$And', '$ArgY', '$And', '$There', '$Is', '$AtMost', '"4"', '$Word', '$Between', '$ArgX', '$And', '$ArgY'],
        ['$The', '$Word', '"isSPACEbasedSPACEin"', '$Is', '$Between', '$ArgX', '$And', '$ArgY', '$And', '$There', '$Is', '$AtMost', '"5"', '$Word', '$Between', '$ArgX', '$And', '$ArgY'],
        ['$The', '$Word', '"thenSPACEknownSPACEas"', '$Is', '$Between', '$ArgX', '$And', '$ArgY', '$And', '$There', '$Is', '$AtMost', '"6"', '$Word', '$Between', '$ArgX', '$And', '$ArgY'],
        ['$ArgX', '$And', '$ArgY', '$SandWich', '$The', '$Word', '"PUNCT5sSPACEmother"', '$And', '$There', '$Is', '$AtMost', '"3"', '$Word', '$Between', '$ArgX', '$And', '$ArgY'],
        ['$There', '$Is', '$AtMost', '"4"', '$Word', '$Between', '$ArgX', '$And', '$ArgY', '$And', '$ArgX', '$And', '$ArgY', '$SandWich', '$The', '$Word', '"secretlySPACEmarried"']
    ]

    correct_parses = [
        '\'@And\'(\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("3",\'tokens\'))),\'@Is\'(\'@Word\'("PUNCT5sSPACEdaughter"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))))',
        '\'@And\'(\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("3",\'tokens\'))),\'@Is\'(\'@Word\'("wasSPACEborn"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))))',
        '\'@And\'(\'@Is\'("asSPACEpartSPACEofSPACEits",\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))),\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("5",\'tokens\'))))',
        '\'@And\'(\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("5",\'tokens\'))),\'@Is\'(\'@Word\'("whoSPACEdiedSPACEof"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))))',
        '\'@And\'(\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("4",\'tokens\'))),\'@Is\'(\'@Word\'("isSPACEfrom"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))))',
        '\'@And\'(\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("4",\'tokens\'))),\'@Is\'(\'@Word\'("PUNCT5sSPACEgrandmother"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))))',
        '\'@And\'(\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("5",\'tokens\'))),\'@Is\'(\'@Word\'("isSPACEbasedSPACEin"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))))',
        '\'@And\'(\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("6",\'tokens\'))),\'@Is\'(\'@Word\'("thenSPACEknownSPACEas"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))))',
        '\'@And\'(\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("3",\'tokens\'))),\'@Is\'(\'@Word\'("PUNCT5sSPACEmother"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))))',
        '\'@And\'(\'@Is\'(\'@Word\'("secretlySPACEmarried"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))),\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("4",\'tokens\'))))'
    ]

    parses = []
    for tokenization in re_tokens:
        parses.append(utils.parse_tokens(tokenization, parser_grammar))
    
    for i, c_parse in enumerate(correct_parses):
        assert c_parse in parses[i]

def test_parse_tokens_ec():
    ec_ccg_trainer = CCGParserTrainer(task="ec", explanation_file="data/ec_test_data.json",
                                      unlabeled_data_file="data/carer_test_data.json")
    explanation_file = ec_ccg_trainer.params["explanation_file"]
    ec_ccg_trainer.load_data(explanation_file)
    parser = ec_ccg_trainer.parser
    parser.create_and_set_grammar()
    parser_grammar = parser.grammar
    
    ec_tokens = [
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"angry"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"increasingSPACEanger"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"beenSPACEsoSPACEexcited"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"whenSPACEtheSPACEgtaSPACEonlineSPACEbikerSPACEdlcSPACEcomesSPACEout"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"bigot"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"howSPACEshit"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"terrorSPACEthreat"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"panic"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"smiling"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"cheerySPACEsmile"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"soSPACEsad"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"mySPACEdepression"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"bitSPACEsurprised"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"soSPACEsudden"'],
        ['$The', '$Sentence', '$Contains', '$The', '$Word', '"honesty"'],
        ['$The', '$Word', '"dependentSPACEonSPACEanotherSPACEperson"', '$In', '$The', '$Sentence']
    ]

    correct_parses = [
        '\'@In1\'(\'Sentence\',\'@Word\'("angry"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("increasingSPACEanger"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("beenSPACEsoSPACEexcited"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("whenSPACEtheSPACEgtaSPACEonlineSPACEbikerSPACEdlcSPACEcomesSPACEout"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("bigot"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("howSPACEshit"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("terrorSPACEthreat"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("panic"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("smiling"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("cheerySPACEsmile"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("soSPACEsad"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("mySPACEdepression"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("bitSPACEsurprised"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("soSPACEsudden"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("honesty"))',
        '\'@In1\'(\'Sentence\',\'@Word\'("dependentSPACEonSPACEanotherSPACEperson"))'
    ]

    parses = []
    for tokenization in ec_tokens:
        parses.append(utils.parse_tokens(tokenization, parser_grammar))
    
    for i, c_parse in enumerate(correct_parses):
        assert c_parse in parses[i]

def test_create_semantic_repr():
    parsed_rep = '\'@And\'(\'@Is\'(\'There\',\'@AtMost\'(\'@between\'(\'@And\'(\'ArgY\',\'ArgX\')),\'@Num\'("3",\'tokens\'))),\'@Is\'(\'@Word\'("PUNCT5sSPACEdaughter"),\'@between\'(\'@And\'(\'ArgY\',\'ArgX\'))))'
    
    semantic_rep = ('.root',
                        ('@And',
                            ('@Is',
                                'There',
                                ('@AtMost',
                                    ('@between', ('@And', 'ArgY', 'ArgX')),
                                    ('@Num', '3', 'tokens'))),
                            ('@Is', ('@Word', "'s daughter"), ('@between', ('@And', 'ArgY', 'ArgX')))))

    assert semantic_rep == utils.create_semantic_repr(parsed_rep)

def test_check_clauses_in_parse_filter():
    semantic_counts = {
        ('.root',
            ('@And',
                ('@Is',
                    ('@And', 'There', 'ArgY'),
                    ('@AtMost',
                        ('@between', ('@And', 'ArgY', 'ArgX')),
                        ('@Num', '3', 'tokens'))),
                ('@Is', ('@Word', "'s daughter"), ('@between', 'ArgX')))): 1,
        ('.root',
            ('@And',
                ('@Is',
                    'There',
                    ('@AtMost',
                        ('@between', ('@And', 'ArgY', 'ArgX')),
                        ('@Num', '3', 'tokens'))),
                ('@Is',
                    ('@Word', "'s daughter"),
                    ('@between', ('@And', 'ArgY', 'ArgX'))))): 1
    }

    final_counts = {
        ('.root',
            ('@And',
                ('@Is',
                    'There',
                    ('@AtMost',
                        ('@between', ('@And', 'ArgY', 'ArgX')),
                        ('@Num', '3', 'tokens'))),
                ('@Is',
                    ('@Word', "'s daughter"),
                    ('@between', ('@And', 'ArgY', 'ArgX'))))): 1
    }

    filtered_counts = utils.check_clauses_in_parse_filter(semantic_counts)

    assert filtered_counts == final_counts

    semantic_counts = {
        ('.root',
            ('@Is',
                'There',
                ('@WordCount',
                    ('@Num', '1', 'head'),
                    ('@Word', 'head'),
                    ('@between', ('@And', 'ArgY', 'ArgX'))))): 1,
        ('.root',
            ('@Is',
                'There',
                ('@WordCount',
                    ('@Num', '1', 'tokens'),
                    'head',
                    ('@between', ('@And', 'ArgY', 'ArgX'))))): 1,
        ('.root',
            ('@Is',
            '   There',
                ('@WordCount',
                    ('@Num', '1', ('@Word', 'head')),
                    ('@And', 'ArgY', 'ArgX'),
                    ('@between', ('@And', 'ArgY', 'ArgX'))))): 1
    }

    final_counts = {
        ('.root',
            ('@Is',
                'There',
                ('@WordCount',
                    ('@Num', '1', 'tokens'),
                    'head',
                    ('@between', ('@And', 'ArgY', 'ArgX'))))): 1
    }

    filtered_counts = utils.check_clauses_in_parse_filter(semantic_counts)

    assert filtered_counts == final_counts

    semantic_counts = {
        ('.root',
            ('@And',
                ('@Is',
                    'There',
                    ('@AtMost',
                        ('@between', ('@And', 'ArgY', 'ArgX')),
                        ('@Num', '3', 'tokens'))),
                ('@Is',
                    ('@Word', "'s daughter"),
                    ('@between', ('@And', 'ArgY', 'ArgX'))))): 1,
        ('.root',
            ('@And',
                ('@Is',
                    'There',
                    ('@AtMost',
                        ('@between', ('@And', 'ArgX', 'ArgY')),
                        ('@Num', '3', 'tokens'))),
                ('@Is',
                    ('@Word', "'s daughter"),
                    ('@between', ('@And', 'ArgX', 'ArgY'))))): 1
    }

    final_counts = {
        ('.root',
            ('@And',
                ('@Is',
                    'There',
                    ('@AtMost',
                        ('@between', ('@And', 'ArgY', 'ArgX')),
                        ('@Num', '3', 'tokens'))),
                ('@Is',
                    ('@Word', "'s daughter"),
                    ('@between', ('@And', 'ArgY', 'ArgX'))))): 1
    }


    filtered_counts = utils.check_clauses_in_parse_filter(semantic_counts)

    assert filtered_counts == final_counts

    semantic_counts = {
        ('.root', ('@In1', 'Sentence', ('@Word', 'increasing anger'))): 1
    }

    filtered_counts = utils.check_clauses_in_parse_filter(semantic_counts)

    assert filtered_counts == semantic_counts

def test_create_labeling_function_sa():
    semantic_reps = [
        ('.root', ('@In1', 'Sentence', ('@Word', 'angry'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'increasing anger'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'been so excited'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'when the gta online biker dlc comes out'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'bigot'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'how shit'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'terror threat'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'panic'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'smiling'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'cheery smile'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'so sad'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'my depression'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'bit surprised'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'so sudden'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'honesty'))),
        ('.root', ('@In1', 'Sentence', ('@Word', 'dependent on another person')))
    ]

    for rep in semantic_reps:
        assert utils.create_labeling_function(rep) != False

def test_create_labeling_function_re():
    # to ensure parity with original paper code, author gave semantic reps
    # and just ensuring the create label works on all of them.
    with open("data/semantic_reps_re.p", "rb") as f:
        re_semantic_reps = pickle.load(f)
    
    for rep in re_semantic_reps:
        assert utils.create_labeling_function(rep) != False
    
    # and a bit more of a visual of what's going on
    semantic_reps = [
        ('.root',
            ('@And',
            ('@Is',
                'There',
                ('@AtMost',
                ('@between', ('@And', 'ArgY', 'ArgX')),
                ('@Num', '3', 'tokens'))),
            ('@Is', ('@Word', "'s daughter"), ('@between', ('@And', 'ArgY', 'ArgX'))))),
        ('.root',
            ('@And',
            ('@Is',
                'There',
                ('@AtMost',
                ('@between', ('@And', 'ArgY', 'ArgX')),
                ('@Num', '3', 'tokens'))),
            ('@Is', ('@Word', 'was born'), ('@between', ('@And', 'ArgY', 'ArgX'))))),
        ('.root',
            ('@And',
            ('@Is', 'as part of its', ('@between', ('@And', 'ArgY', 'ArgX'))),
            ('@Is',
                'There',
                ('@AtMost',
                ('@between', ('@And', 'ArgY', 'ArgX')),
            ('@Num', '5', 'tokens'))))),
        ('.root',
            ('@And',
            ('@Is',
                'There',
                ('@AtMost',
                ('@between', ('@And', 'ArgY', 'ArgX')),
                ('@Num', '5', 'tokens'))),
            ('@Is', ('@Word', 'who died of'), ('@between', ('@And', 'ArgY', 'ArgX'))))),
        ('.root',
            ('@And',
            ('@Is',
                'There',
                ('@AtMost',
                ('@between', ('@And', 'ArgY', 'ArgX')),
                ('@Num', '4', 'tokens'))),
            ('@Is', ('@Word', 'is from'), ('@between', ('@And', 'ArgY', 'ArgX')))))
    ]

    for rep in semantic_reps:
        assert utils.create_labeling_function(rep) != False