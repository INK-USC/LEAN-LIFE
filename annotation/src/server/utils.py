import string
from .constants import EXPLANATION_SETTINGS_MAP, NAMED_ENTITY_RECOGNITION_VALUE, RELATION_EXTRACTION_VALUE, SENTIMENT_ANALYSIS_VALUE
import spacy
import re

class LeanLifeSpacyWrapper():
    bad_noun_phrases = set(["don't"])
    def __init__(self, spacy_model="en"):
        self.nlp = spacy.load(spacy_model)
        rules = {}
        apostrophes = ["'", "'", "'", "’", "'"]
        for key , value in self.nlp.tokenizer.rules.items():
            add = True
            for a in apostrophes:
                if a in key:
                    add = False
                    break
            if add:
                rules[key] = value

        self.nlp.tokenizer.rules = rules
    
    def tokenize(self, text):
        doc = self.nlp(text)
        final_tokens = []
        i = -1
        for token in doc:
            token_text = token.text
            if "'" == token_text[0]:
                final_tokens[i] = final_tokens[i]+token_text
            elif "––" in token.text:
                split = token_text.split("––")
                if len(split[0]) > 1:
                    i += 1
                    final_tokens.append(split[0])
                    i += 1
                    final_tokens.append("––")
                    i += 1
                    final_tokens.append(split[1])
            else:
                i += 1
                final_tokens.append(token_text)
        
        return final_tokens

    def complete_paranthese(self, phrase):
        if "(" in phrase and ")" not in phrase:
            phrase = phrase + ")"
            return phrase
        elif "{" in phrase and "}" not in phrase:
            phrase = phrase + "}"
            return phrase
        elif "[" in phrase and "]" not in phrase:
            phrase = phrase + "]"
            return phrase
        
        return phrase
        
    def get_char_offsets_of_noun_phrase(self, noun_phrase, text, lower=True):
        matches = []
        noun_phrase = self.complete_paranthese(noun_phrase)
        if lower:
            noun_phrase = noun_phrase.lower()
            text = text.lower()
        for match in re.finditer(noun_phrase, text):
            start_offset = match.start()
            end_offset = match.end()
            if start_offset == 0:
                if text[end_offset] in string.punctuation or text[end_offset] == " ":
                    matches.append({"start_offset" : start_offset, "end_offset" : end_offset})
            elif end_offset == len(text):
                if text[start_offset-1] in string.punctuation or text[start_offset-1] == " ":
                    matches.append({"start_offset" : start_offset, "end_offset" : end_offset})
            else:
                start_valid = (text[start_offset-1] in string.punctuation or text[start_offset-1] == " ")
                end_valid = (text[end_offset] in string.punctuation or text[end_offset] == " ")
                if start_valid and end_valid:
                    matches.append({"start_offset" : start_offset, "end_offset" : end_offset})
        
        return matches

    def get_noun_phrases(self, text):
        doc = self.nlp(text)
        noun_phrases = []
        noun_chunks = list(set([chunk for chunk in doc.noun_chunks]))
        for chunk in noun_chunks:
            if chunk.text not in self.bad_noun_phrases:
                char_offsets = self.get_char_offsets_of_noun_phrase(chunk.text, text)
                for offset in char_offsets:
                    noun_phrases.append(offset)
        return noun_phrases
             
    def __call__(self, text):
        doc = self.nlp(text)
        return doc

SPACY_WRAPPER = LeanLifeSpacyWrapper()
        
def get_key_choices():

    selectKey, shortKey = [c for c in string.ascii_lowercase], [c for c in string.ascii_lowercase]
    checkKey = 'ctrl shift'
    shortKey += [ck + ' ' + sk for ck in checkKey.split() for sk in selectKey]
    shortKey += [checkKey + ' ' + sk for sk in selectKey]
    shortKey += ['']
    KEY_CHOICES = ((u, c) for u, c in zip(shortKey, shortKey))
    return KEY_CHOICES

def convert_task_name_to_annotation_type(task_name):
    if task_name == NAMED_ENTITY_RECOGNITION_VALUE:
        annotation_type = "named_entity_annotation"
    elif task_name == RELATION_EXTRACTION_VALUE:
        annotation_type = "relation_extraction_annotation"
    elif task_name == SENTIMENT_ANALYSIS_VALUE:
        annotation_type = "sentiment_analysis_annotation"
    
    return annotation_type

def convert_explanation_int_to_explanation_type(explanation_int):
    return EXPLANATION_SETTINGS_MAP[explanation_int]
    
