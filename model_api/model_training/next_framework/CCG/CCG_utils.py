"""
    Functions that are used in the pipeline to convert explanations into labeling functions
"""
import copy
import pathlib
import re
import string
import sys
from nltk.ccg import chart, lexicon
PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
from CCG import CCG_constants as constants
from CCG import CCG_util_classes as util_classes

def _find_quoted_phrases(explanation):
    """
        Checks for the existence of a quoted phrase within an explanation
        Three types of quotes are accepted
        
        Arguments:
            explanation (str) : explanation text for a labeling decision
        
        Returns:
            arr : an array of quoted phrases or an empty array
    """
    possible_queries = re.findall('"[^"]+"', explanation)
    if len(possible_queries):
        possible_queries = [query for query in possible_queries]
        return possible_queries
    
    possible_queries = re.findall("'[^']+'", explanation)
    if len(possible_queries):
        possible_queries = [query for query in possible_queries]
        return possible_queries

    possible_queries = re.findall("`[^`]+`", explanation)
    if len(possible_queries):
        possible_queries = [query for query in possible_queries]
        return possible_queries
    
    return []

def segment_explanation(explanation):
    """
        Segments an explanation into portions that have a quoted phrase and those that don't
        Prepends the quoted word segments with an asterisk to indicate the existences of a quoted phrase
        Arguments:
            explanation (str) : raw explanation text
        
        Returns:
            arr : segmented explanation
    """
    possible_queries = _find_quoted_phrases(explanation)
    pieces = []
    last_end_position = 0
    for query in possible_queries:
        start_position = explanation.find(query)
        piece = explanation[last_end_position:start_position]
        last_end_position = start_position + len(query)
        pieces.append(piece)
        pieces.append("*"+query)
    
    if last_end_position < len(explanation):
        pieces.append(explanation[last_end_position:])
    
    return pieces

def clean_text(text, lower=True, collapse_punc=True, commas=True, switch_amp=True, exclusive=True, whitespace=True):
    """
        Function that cleans text in the following way:
            1. Lowecases the text
            2. Replaces all !.? with a a single ., keeps any ... in the text
            3. Replaces multiple commas with a single comma (same for ":")
            4. Removes all non [lowercase characters, \., \,, digit, \', \", >, <, =, \/]
            5. Removes unnecessary whitespace within text 
        
        Used for Cleaning Explanations.
        Arguments:
            text           (str) : text to be cleaned
            lower         (bool) : lowercase text or not
            collapse_punc (bool) : whether to collapse punctuation to a single .
            commas        (bool) : whether to get rid of unnecessary commas
            switch_amp    (bool) : whether to switch & into 'and'
            exclusive     (bool) : remove all characters not mentioned above
            whitespace    (bool) : whether to get rid of unnecessary whitespace
        
        Returns:
            str : cleaned text
    """
    if lower:
        text = text.lower()
    if collapse_punc:
        text = re.sub(r'\?+', '.', text)
        text = re.sub(r'\!+', '.', text)
        text = re.sub(r'(\.){2}', ".", text)
        text = re.sub(r'(\.){4,}', ".", text)
    if commas:
        text = re.sub(r",{2,}", ",", text)
    if switch_amp:
        text = re.sub(r'&', 'and', text)
    if exclusive:
        text = re.sub(r"[^a-z .,0-9'\"><=\/]", "", text)
    if whitespace:
        text = " ".join(text.split()).strip()

    return text

def clean_explanation(explanation):
    """
        Function that cleans an explanation text, but also makes sure to leave text within quotes uncleaned
        Arguments:
            explanation (str) : explanation to be cleaned
        
        Returns:
            str : cleaned text
    """
    segments = segment_explanation(explanation)
    for i, segment in enumerate(segments):
        if len(segment):
            if segment[0] != "*":
                segments[i] = clean_text(segment)
            else:
                segments[i] = "\"" + segment[2:len(segment)-1].lower() + "\""
    
    cleaned_explanation = " ".join(segments).strip()

    return cleaned_explanation

def convert_chunk_to_terminal(chunk):
    """
        Given a chunk we have three possible ways to convert it to a terminal in our lexicon.
        1. Check for chunk in our lexicon and convert it
        2. If a chunk matches text from the original statement, indicated in the explanation by " ", 
           then consider that its own terminal
        3. If its a number, we put quotes around it
        4. If its a numeric string we convert its base 10 form and put quotes around it
        5. Else this chunk will not be matched to a terminal
        Arguments:
            chunk (str) : chunk that needs to be converted
        
        Returns:
            arr|None : array of terminals that this chunk can converted to
                       if no good match is found None is returned
    """
    num_lexicon = ['one','two','three','four','five','six','seven','eight','nine', 'ten']
    num_lexicon_dict = {elem:str(i+1) for i,elem in enumerate(num_lexicon)}
    if chunk in constants.TOKEN_TERMINAL_MAP:
        return constants.TOKEN_TERMINAL_MAP[chunk]
    elif chunk.startswith("\"") and chunk.endswith("\"") or chunk in constants.PHRASE_VALUES:
        return [chunk]
    elif chunk.isdigit():
        return ["\""+chunk+"\""]
    elif chunk in num_lexicon:
        return ["\""+num_lexicon_dict[chunk]+"\""]
    else:
        return None

def chunk_explanation(cleaned_explanation, nlp):
    """
        Taking a cleaned explanation, we then chunk it to allow it to tokens to be converted
        into predicates from our grammar's lexicon. We use spaCy to tokenize the model, but also:
            1. check for certain phrases and convert them automatically into our lexicon space
            2. capture phrases in quotes and keep them as one "token"
        
        Arguments:
            cleaned_explanation (str) : output of clean_explanation
            nlp         (spaCy model) : an instantiated spaCy model
        
        Returns:
            arr : chunks (str) that can be converted into predicates of our grammar's vocab
    """
    for i, phrase in enumerate(constants.PHRASE_ARRAY):
        if phrase in cleaned_explanation:
            temp_sub = constants.PHRASE + str(i)
            cleaned_explanation = re.sub(phrase, temp_sub, cleaned_explanation)
    
    doc = nlp(cleaned_explanation)
    initial_tokens = [token.text for token in doc]
    final_tokens = []
    cur_token = ""
    inside = False
    for i, token in enumerate(initial_tokens):
        if token.startswith("\"") and not inside:
            if token.endswith("\"") and len(token) > 1:
                final_tokens.append(token)
            else:
                cur_token = token
                inside = True
        elif token.endswith("\""):
            if token != "\"":
                cur_token += " "
            cur_token += token
            final_tokens.append(cur_token)
            inside = False
        elif inside:
            if cur_token != "\"":
                cur_token += " "
            cur_token += token
        else:
            final_tokens.append(token)
    
    for i, token in enumerate(final_tokens):
        if token.startswith(constants.PHRASE):
            phrase = constants.PHRASE_ARRAY[int(token.split(constants.PHRASE)[1])]
            final_tokens[i] = constants.PHRASE_TERMINAL_MAP[phrase]
    return final_tokens

def clean_and_chunk(explanation, nlp):
    """
        Wrapper function
    """
    return chunk_explanation(clean_explanation(explanation), nlp)

def prepare_token_for_rule_addition(token, reverse=False):
    """
        Certain tokens spawn new rules in the CCG grammar, before the rule can be created though
        we must normalize the token. We do this to ensure the parsing of the grammar by the CCGChartParser.
        We also reverse the normalization when needed, but collapse certain patterns to a single comma.
        Arguments:
            token    (str) : token to be normalized
            reverse (bool) : reverse the normalization
        
        Returns:
            str : (un)normalized token
    """
    punctuations = string.punctuation.replace("\"", "")
    if reverse:
        token = token.replace(constants.SPACE, ' ')
        for i in range(len(punctuations)-1, -1, -1):
            replacement = constants.PUNCT + str(i)
            token = token.replace(replacement, punctuations[i])
        return token
    
    token = token.replace(' ', constants.SPACE)
    for i, punc in enumerate(punctuations):
        replacement = constants.PUNCT + str(i)
        token = token.replace(punc, replacement)
    return token

def add_rules_to_grammar(tokens, grammar_string):
    """
        Certain tokens require rules to be addeed to a base CCG grammar, depending on the token
        this function adds those rules to the grammar.
        Arguments:
            tokens         (arr) : tokens that require new rules to be added to the grammar
            grammar_string (str) : string representation of the base grammar to add to
        Returns:
            str : updated grammar_string
    """
    grammar = grammar_string
    for token in tokens:
        raw_token = token[1:len(token)-1]
        token = prepare_token_for_rule_addition(token)
        if raw_token.isdigit():
            grammar = grammar + "\n\t\t" + token + " => NP/NP {\\x.'@Num'(" + token + ",x)}" + "\n\t\t" + token + " => N/N {\\x.'@Num'(" + token + ",x)}"+"\n"
            grammar = grammar + "\n\t\t" + token + " => NP {" + token + "}" + "\n\t\t" + token + " => N {" + token + "}"+"\n"
            grammar = grammar + "\n\t\t" + token + " => PP/PP/NP/NP {\\x y F.'@WordCount'('@Num'(" + token + ",x),y,F)}" + "\n\t\t" + token + " => PP/PP/N/N {\\x y F.'@WordCount'('@Num'(" + token + ",x),y,F)}"+"\n"
        else:
            grammar = grammar + "\n\t\t" + token + " => NP {"+token+"}"+"\n\t\t"+token+" => N {"+token+"}"
    return grammar

def generate_phrase(sentence, nlp):
    """
        Generate a useful wrapper object for each sentence in the data
        Arguments:
            sentence    (str) : sentence to generate wrapper for
            nlp (spaCy model) : pre-loaded spaCy model to use for NER detection
        
        Returns:
            Phrase : useful wrapper object
    """
    if "SUBJ" in sentence and "OBJ" in sentence:
        subj_type = re.search(r"SUBJ-[A-Z_'s,]+", sentence).group(0).split("-")[1].strip()
        subj_type = subj_type.replace("'s", "")
        subj_type = subj_type.replace(",", "")
        obj_type = re.search(r"OBJ-[A-Z_'s,]+", sentence).group(0).split("-")[1].strip()
        obj_type = obj_type.replace("'s", "")
        obj_type = obj_type.replace(",", "")
        sentence = re.sub(r"SUBJ-[A-Z_]+", "SUBJ", sentence)
        sentence = re.sub(r"SUBJ-[A-Z_'s]+", "SUBJ's", sentence)
        sentence = re.sub(r"SUBJ-[A-Z_]+,", "SUBJ,", sentence)
        sentence = re.sub(r"OBJ-[A-Z_]+", "OBJ", sentence)
        sentence = re.sub(r"OBJ-[A-Z_'s]+", "OBJ's ", sentence)
        sentence = re.sub(r"OBJ-[A-Z_]+,", "OBJ,", sentence)

    doc = nlp(sentence)
    ners = [token.ent_type_ if token.text not in ["SUBJ", "OBJ"] else "" for token in doc]
    tokens = [token.text.lower() for token in doc]
    # soft matching functions depend on these values, so if no SUBJ or OBJ exist
    # I want the function to error, as an inappropriate explanation was created
    # for the dataset being parsed, hence 2*len(tokens) as tokens[subj_posi] will always error
    subj_posi = 2*len(tokens)
    obj_posi = 2*len(tokens)
    indices_to_pop = []
    for i, token in enumerate(tokens):
        if token == "subj":
            if subj_posi < len(tokens):
                indices_to_pop.append(i)
            else:
                subj_posi = i
                ners[i] = subj_type
        elif token == "obj":
            if obj_posi < len(tokens):
                indices_to_pop.append(i)
            else:
                obj_posi = i
                ners[i] = obj_type

    if len(indices_to_pop):
        indices_to_pop.reverse()
        for index in indices_to_pop:
            ners.pop(index)
            tokens.pop(index)
    
    return util_classes.Phrase(tokens, ners, subj_posi, obj_posi)

def parse_tokens(one_sent_tokenize, raw_lexicon):
    """
        CYK algorithm for parsing a tokenized sentence into a parse tree. We implement our own, as solely
        using NLTK's CCGChartParser and the grammar we came up won't allow for the parses we desired. As
        we are not linguists, we found it easier to change the code than figure out possible problems with
        our grammar.

        Outputs the last row of the CYK datastructure as possible parses for the sentence
            * Each element in the row is string version of nltk.tree.Tree (sort of, we actually construct our
              own tree based on the tree provided by NLTK)

        Arguments:
            one_sent_tokenize (arr) : array of string tokens representing a sentence
            raw_lexicon       (str) : string representation of lexicon (grammar and vocabulary rep of a language)
        
        Returns:
            (arr) : list of possible parses, read comment above for more
    """
    try:
        beam_lexicon = copy.deepcopy(raw_lexicon)
        CYK_form = [[[token] for token in one_sent_tokenize]]
        CYK_sem = [[]]
        for layer in range(1,len(one_sent_tokenize)):
            layer_form = []
            layer_sem = []
            lex = lexicon.fromstring(beam_lexicon, True)
            parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
            for col in range(0,len(one_sent_tokenize)-layer):
                form = []
                sem_temp = []
                word_index = 0
                st = col+0
                ed = st+layer
                for splt in range(st,ed):
                    words_L = CYK_form[splt-st][st]
                    words_R = CYK_form[ed-splt-1][splt+1]
                    for word_0 in words_L:
                        for word_1 in words_R:
                            try:
                                for parse in parser.parse([word_0, word_1]):
                                    (token, op) = parse.label()
                                    categ = token.categ()
                                    sem = token.semantics()
                                    word_name = '$Layer{}_Horizon{}_{}'.format(str(layer), str(col),str(word_index))
                                    word_index+=1
                                    entry = "\n\t\t"+word_name+' => '+str(categ)+" {"+str(sem)+"}"
                                    if str(sem)+'_'+str(categ) not in sem_temp:
                                        form.append((parse,word_name,entry,str(sem)))
                                        sem_temp.append(str(sem)+'_'+str(categ))
                            except:
                                pass
                add_form = []
                for elem in form:
                    parse, word_name, entry,sem_ = elem
                    add_form.append(word_name)
                    beam_lexicon = beam_lexicon+entry
                    layer_sem.append(sem_)
                layer_form.append(add_form)
            CYK_form.append(layer_form)
            CYK_sem.append(layer_sem)
        return CYK_sem[-1]
    except:
        return []

def create_semantic_repr(semantic_rep):
    """
        Given a semtantic string representation of a parse tree we transform the tree into a hierarchical
        structure, so that a conversion to a labeling function is possible; functions at the top rely on the 
        return value of functions lower down.
        In doing this, we loose the original lexical heirachy of the parse tree.
        
        This differs from this approach https://homes.cs.washington.edu/~lsz/papers/zc-uai05.pdf
        We can't create features based on lexical structure, but only semantic structure
        If the semantic string representation of the tree makes sense when considering the ordering
        of the functions making up the string, then we output the hierarchical tuple, else we return
        false. False here indicates that while a valid parse tree was attainable, the semantics of the
        parse do not make sense though.

        Arguments:
            semantic_rep (str) :  tree representation of tree from our parse_tokens function
        
        Returns:
            tuple | false : if valid semantically, we output a tuple describing the semantics, else false
    """
    clauses = re.split(',|(\\()',semantic_rep)
    delete_index = []
    for i in range(len(clauses)-1, -1, -1):
        if clauses[i] == None:
            delete_index.append(i)
    for i in delete_index:
        del clauses[i]
        
    # Switch poisition of ( and Word before it
    switched_semantics = []
    for i, token in enumerate(clauses):
        if token=='(':
            switched_semantics.insert(-1,'(')
        else:
            switched_semantics.append(token)
    
    # Converting semantic string into a multi-level tuple, ex: (item, tuple) would be a two level tuple
    # This representation allows for the conversion from semantic representation to labeling function
    hierarchical_semantics = ""
    for i, clause in enumerate(switched_semantics):
        prepped_clause = clause
        if prepped_clause.startswith("\""):
            prepped_clause = prepare_token_for_rule_addition(prepped_clause, reverse=True)
            if prepped_clause.endswith(")"):
                posi = len(prepped_clause)-1
                while prepped_clause[posi]==")":
                    posi-=1
                assert prepped_clause[posi]=="\""
            else:
                posi = len(prepped_clause)-1
                assert prepped_clause[posi] == "\""
            prepped_clause = prepped_clause[0] + \
                             prepped_clause[1:posi].replace('\'','\\\'') + \
                             prepped_clause[posi:]

        if switched_semantics[i-1] != "(" and len(hierarchical_semantics):
            hierarchical_semantics += ","

        hierarchical_semantics += prepped_clause
    # if the ordering of the semantics in this semantic representation is acceptable per the functions
    # the semantics map to, then we will be able to create the desired multi-label tuple
    # else we return False
    try:
        hierarchical_tuple = ('.root', eval(hierarchical_semantics))
        return hierarchical_tuple
    except:
        return False

def _detect_semantic_token(semantic_repr, token):
    """ Exists solely for documentation purposes """
    return token in semantic_repr

def _count_correct_between_parses(semantic_repr):
    """
        One flaw in the grammar is the parsing of between statements, so this function counts the number
        of correct between statements in a parse.

        Returns:
            semantic_repr (str) : str rep of semantic representation of a parse (output of create_semantic_repr)
        
        Returns:
            int : number of correct between parses
    """
    correct_between_clauses = ["('@between', ('@And', 'ArgY', 'ArgX'))", 
                               "('@between', ('@And', 'ArgX', 'ArgY'))"]
        
    count = semantic_repr.count(correct_between_clauses[0]) + \
            semantic_repr.count(correct_between_clauses[1])
    
    return count

def _count_correct_num_parses(semantic_repr):
    """
        Another flaw in the grammar is the parsing of counting the number of tokens between phrases and/or
        anchor words, so this function counts the number of correct "number statements" in a parse.

        Returns:
            semantic_repr (str) : str rep of semantic representation of a parse (output of create_semantic_repr)
        
        Returns:
            int : number of correct number parses
    """
    num_regex = r"\('\@Num', '[0-9]+', 'tokens'\)"
    count = len(re.findall(num_regex, semantic_repr))
    
    return count

def check_clauses_in_parse_filter(semantic_counts):
    """
        The parse function sometimes creates incorrect parses due to problems in our grammar. For at least
        the most common errors in our grammars, between and Num clauses, we check for the number of correct
        clause parses within a larger parse and select the larger parse with the max number. We only do this
        filtering if we detect the existence of between or Num clauses. In the event of a tie we just take
        the first seen one.

        Returns:
            semantic_counts (dict) : tuple - semantic representation of a parse (output of create_semantic_repr),
                                     value - number of times the parse was produced by our parse function

        Returns:
            dict :  key - string version of the semantic representation of a parse,
                    value - number of times the parse was produced by our parse function
                    ^- filtered though if applicable
    """
    contains_between_clause = False
    contains_num_clause = False
    for key in semantic_counts:
        key_str = str(key)
        contains_between_clause = _detect_semantic_token(key_str, "@between")
        contains_num_clause = _detect_semantic_token(key_str, "@Num")
        if contains_between_clause or contains_num_clause:
            break 
    
    if contains_between_clause or contains_num_clause:
        max_correct_count = 0
        correct_semantic_rep = None
        for key in semantic_counts:
            count = 0
            key_str = str(key)
            if contains_between_clause:
                count += _count_correct_between_parses(key_str)
            if contains_num_clause:
                count += _count_correct_num_parses(key_str)
            if count > max_correct_count:
                correct_semantic_rep = {key : semantic_counts[key]}
                max_correct_count = count
        
        return correct_semantic_rep
    
    return semantic_counts

def create_labeling_function(semantic_repr, level=0):
    """
        Creates a labeling function (lambda function) from a hierarchical tuple representation
        of the semantics of a parse tree. The labeling function takes in a Phrase object and then
        evaluates whether the labeling function applies to this Phrase object.
        Arguments:
            semantic_repr (tuple) : hierarchical tuple representation
        
        Returns:
            function | false : if a function is creatable via the tuple, it is created, else false
    """
    try:
        if isinstance(semantic_repr, tuple):
            op = constants.STRICT_MATCHING_OPS[semantic_repr[0]]
            args = [create_labeling_function(arg, level=level+1) for arg in semantic_repr[1:]]
            if False in args:
                return False
            return op(*args) if args else op
        else:
            if semantic_repr in constants.NER_TERMINAL_TO_EXECUTION_TUPLE:
                return constants.NER_TERMINAL_TO_EXECUTION_TUPLE[semantic_repr]
            else:
                return semantic_repr
    except:
        return False

def create_soft_labeling_function(semantic_repr, level=0):
    """
        Creates a labeling function (lambda function) from a hierarchical tuple representation
        of the semantics of a parse tree. The labeling function takes in a Phrase object and then
        evaluates whether the labeling function applies to this Phrase object.
        Arguments:
            semantic_repr (tuple) : hierarchical tuple representation
        
        Returns:
            function | false : if a function is creatable via the tuple, it is created, else false
    """
    try:
        if isinstance(semantic_repr, tuple):
            op = constants.SOFT_MATCHING_OPS[semantic_repr[0]]
            args = [create_soft_labeling_function(arg) for arg in semantic_repr[1:]]
            if False in args:
                return False
            return op(*args) if args else op
        else:
            if semantic_repr in constants.NER_TERMINAL_TO_EXECUTION_TUPLE:
                return constants.NER_TERMINAL_TO_EXECUTION_TUPLE[semantic_repr]
            else:
                return semantic_repr
    except:
        return False