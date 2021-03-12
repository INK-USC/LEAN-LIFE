import pdb
def tokens_to_string(tokens):
    return " ".join(tokens)

class DataPoint():
    """
        Class that ties together some useful properties we need to store about a datapoint (x)

        Attributes:
            sentence             (Phrase) : a wrapper object around the original text of the datapoint
            raw_explanation         (str) : if an explanation is present we store it for further processing
            chunked_explanation    (list) : a list containing chunks of text that can be converted into 
                                            predicates certain chunks are already converted to tokens to 
                                            simplify the tokenizing process.
            tokenized_explanations (list) : given a list of chunks, multiple possible token sequences 
                                            can be created. Each item in this list is itself a list, each 
                                            inner list is a list of tokens from our grammar
            semantic_counts        (dict) : for each possible sequence of tokens we store the parsed semantic
                                            representation of the raw_explanation. We first convert possible token
                                            sequences into trees and extract the hierarchical semantics tied to each
                                            tree. We then store the unique semantics as keys and count how often each
                                            representation appears.
            labeling_functions     (dict) : key - semantic representation, value - lambda function equivalent
    """
    def __init__(self, sentence, label, explanation=""):
        self.sentence = sentence
        self.label = label
        self.raw_explanation = explanation
        self.chunked_explanation = None
        self.tokenized_explanations = None
        self.semantic_counts = None
        self.labeling_functions = None

    def __repr__(self):
        return str(self.__dict__)

class Phrase():
    """
        Wrapper class taken from original NExT code. Has some functions that are useful for
        explanations that deal with relative position of words w.r.t subj and obj.

        For now mostly untouched, other than guarding against the possability of no subj or obj
            - as is the case for Text Classification
    """
    def __init__(self, tokens, ners, subj_posi, obj_posi):
        self.tokens = tokens
        self.ners = ners
        self.subj_posi = subj_posi
        self.obj_posi = obj_posi
        self.sentence = tokens_to_string(tokens)
        if subj_posi < len(self.tokens):
            self.subj = self.tokens[self.subj_posi]
        else:
            self.subj = ""
        if obj_posi < len(self.tokens):
            self.obj = self.tokens[self.obj_posi]
        else:
            self.obj = ""
    
    def get_mid(self):
        if self.subj_posi != None and self.obj_posi != None:
            st = min(self.subj_posi,self.obj_posi)+1
            ed = max(self.subj_posi,self.obj_posi)
            midphrase = tokens_to_string(self.tokens[st:ed])
            midners = self.ners[st:ed]
            # rename word to phrase
            return {'word':midphrase,'NER':midners,'tokens':self.tokens[st:ed],'position':(st,ed)}

        return {'word':tokens_to_string(self.tokens),'NER':self.ners,'tokens':self.tokens,'position':(0,len(self.tokens))}
    
    def get_other_posi(self,LoR,XoY):
        assert LoR == 'Left' or LoR == 'Right' or LoR=='Range'
        assert XoY == 'X' or XoY == 'Y'

        if self.subj_posi != None and self.obj_posi != None:
            if XoY == 'X':
                split_posi = self.subj_posi
            else:
                split_posi = self.obj_posi

            if LoR=='Left':
                phrase = tokens_to_string(self.tokens[:split_posi])
                phrase_ner = self.ners[:split_posi]
                phrase_tokens = self.tokens[:split_posi]
                posi = (0,split_posi)
            elif LoR=='Right':
                phrase = tokens_to_string(self.tokens[split_posi+1:])
                phrase_ner = self.ners[split_posi+1:]
                phrase_tokens = self.tokens[split_posi+1:]
                posi = (split_posi+1,len(self.tokens))
            else:
                phrase = self.sentence
                phrase_ner = self.ners
                phrase_tokens = self.tokens
                return {'word':phrase,'NER':phrase_ner,'tokens':phrase_tokens,'POSI':split_posi}

            return {'word':phrase,'NER':phrase_ner,'tokens':phrase_tokens,'position':posi}

        return {'word':tokens_to_string(self.tokens),'NER':self.ners,'tokens':self.tokens,'position':(0,len(self.tokens))}

    def with_(self,XoY,SoE,substring):
        assert XoY == 'X' or XoY == 'Y'
        assert SoE == 'starts' or SoE == 'ends'

        if self.subj_posi != None and self.obj_posi != None:
            if XoY=='X':
                word = self.subj
            else:
                word = self.obj

            if SoE=='starts':
                return word.startswith(substring)
            else:
                return word.endswith(substring)
        return False
    
    def update_tokens_and_ners(self, tokens, ners):
        self.tokens = tokens
        self.ners = ners
        if self.subj_posi != None:
            self.subj = self.tokens[self.subj_posi]
        if self.obj_posi != None:
            self.obj = self.tokens[self.obj_posi]
    
