import sys
sys.path.append("../")
import random
import training.util_functions as func

random_state = 42
random.seed(random_state)

def test_build_vocab():
    custom_vocab = func.build_custom_vocab("tacred", 10)
    actual_custom_vocab = {
        'SUBJ-PERSON': 10, 'OBJ-PERSON': 11, 'SUBJ-ORGANIZATION': 12, 'OBJ-ORGANIZATION': 13, 'SUBJ-DATE': 14,
        'OBJ-DATE': 15, 'SUBJ-NUMBER': 16, 'OBJ-NUMBER': 17, 'SUBJ-TITLE': 18, 'OBJ-TITLE': 19,
        'SUBJ-COUNTRY': 20, 'OBJ-COUNTRY': 21, 'SUBJ-LOCATION': 22, 'OBJ-LOCATION': 23, 'SUBJ-CITY': 24,
        'OBJ-CITY': 25, 'SUBJ-MISC': 26, 'OBJ-MISC': 27, 'SUBJ-STATE_OR_PROVINCE': 28, 'OBJ-STATE_OR_PROVINCE': 29,
        'SUBJ-DURATION': 30, 'OBJ-DURATION': 31, 'SUBJ-NATIONALITY': 32, 'OBJ-NATIONALITY': 33,
        'SUBJ-CAUSE_OF_DEATH': 34, 'OBJ-CAUSE_OF_DEATH': 35, 'SUBJ-CRIMINAL_CHARGE': 36, 'OBJ-CRIMINAL_CHARGE': 37,
        'SUBJ-RELIGION': 38, 'OBJ-RELIGION': 39, 'SUBJ-URL': 40, 'OBJ-URL': 41, 'SUBJ-IDEOLOGY': 42,
        'OBJ-IDEOLOGY': 43
    }
    assert custom_vocab == actual_custom_vocab

def test_find_array_start_position():
    big_array = [1,2,3,4,4,5,6,67,78]
    small_array = [4,4,5,6]

    assert func.find_array_start_position(big_array, small_array) == 3

def test_tokenize():
    text = "HEY I've got some funKy things! \nIsn't it Funny!!     "
    tokens = ['HEY', 'I', "'ve", 'got', 'some', 'funKy', 'things', '!', 'Is', "n't", 'it', 'Funny', '!', '!']
    assert func.tokenize(text) == tokens

    text = "SUBJ-PERSON is my friend, they are a OBJ-OCCUPATION down the street."
    tokens = ['SUBJ-PERSON', 'is', 'my', 'friend', ',', 'they', 'are', 'a', 'OBJ-OCCUPATION', 'down', 'the', 'street', '.']
    assert func.tokenize(text) == tokens

def test_build_vocab():
    train = ["Here are some strings", 
             "Some not so interesting strings!", 
             "Let's make ThEm MORe Intersting?",
             "Yes you can :)",
             "Coolio <::>"]
    
    embedding_name = "glove.6B.50d"

    vocab = func.build_vocab(train, embedding_name, save=False)

    
    tokens_in_order = [
        '<unk>', '<pad>', ':', 'strings', '!', "'s", ':)', '<', '>', '?', 'Coolio', 'Here',
        'Intersting', 'Let', 'MORe', 'Some', 'ThEm', 'Yes', 'are', 'can', 'interesting',
        'make', 'not', 'so', 'some', 'you'
    ]
    
    assert "torchtext.vocab.Vocab" in str(type(vocab))
    assert vocab.itos == tokens_in_order

def test_convert_text_to_tokens():
    train = ["Here are some strings", 
             "Some not so interesting strings!", 
             "Let's make ThEm MORe Intersting?",
             "Yes you can :)",
             "Coolio <::>"]
    
    embedding_name = "glove.6B.50d"

    vocab = func.build_vocab(train, embedding_name, save=False)

    custom_vocab = {
        "However" : 50,
        "going" : 51
    }

    sample_data = ["Let's make ThEm MORe Intersting?",
                   "Some not so interesting strings!", 
                   "However, this one is going to have lots of <unk>s"]
        
    tokenized_data = [[13, 5, 21, 16, 14, 12, 9], 
                      [15, 22, 23, 20, 3, 4], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 8, 0]]
    
    assert func.convert_text_to_tokens(sample_data, vocab, func.tokenize) == tokenized_data

    tokenized_data = [[13, 5, 21, 16, 14, 12, 9], 
                      [15, 22, 23, 20, 3, 4], 
                      [50, 0, 0, 0, 0, 51, 0, 0, 0, 0, 7, 0, 8, 0]]
        
    assert func.convert_text_to_tokens(sample_data, vocab, func.tokenize, custom_vocab) == tokenized_data

def test_extract_queries_from_explanations():
    text = 'First type of "query"'

    assert ['query'] == func.extract_queries_from_explanations(text)

    text = 'Another "type" of "query"'

    assert ['type', 'query'] == func.extract_queries_from_explanations(text)

    text = 'Ideally all explanations will only use "double quote\'s", so we can avoid issues with "\'"'

    assert ['double quote\'s', '\''] == func.extract_queries_from_explanations(text)

    text = "An explanation can use 'single quotes'"

    assert ['single quotes'] == func.extract_queries_from_explanations(text)

    text = "However, there can be some problems with 'apostrophes like, 's'"

    assert ['apostrophes like, '] == func.extract_queries_from_explanations(text)

    text = "We can even handle ''double single quotes too''"

    assert ['double single quotes too'] == func.extract_queries_from_explanations(text)

    text = "Though do \"not\" mix 'quotes'"

    assert ['not'] == func.extract_queries_from_explanations(text)

    text = "Finally we also handle `backticks as quotes`"

    assert ["backticks as quotes"] == func.extract_queries_from_explanations(text)

    text = "No quotes here though, so should be empty"

    assert [] == func.extract_queries_from_explanations(text)