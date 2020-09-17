SENTIMENT_ANALYSIS_KEY = 'SentimentAnalysis'
NAMED_ENTITY_RECOGNITION_KEY = 'NamedEntityRecognition'
RELATION_EXTRACTION_KEY = 'RelationExtraction'
SENTIMENT_ANALYSIS_VALUE = 'Sentiment Analysis'
NAMED_ENTITY_RECOGNITION_VALUE = 'Named Entity Recognition'
RELATION_EXTRACTION_VALUE = 'Relation Extraction'
TASK_CHOICES = (
    (SENTIMENT_ANALYSIS_KEY, SENTIMENT_ANALYSIS_VALUE),
    (NAMED_ENTITY_RECOGNITION_KEY, NAMED_ENTITY_RECOGNITION_VALUE),
    (RELATION_EXTRACTION_KEY, RELATION_EXTRACTION_VALUE),
)

EXPLANATION_CHOICES = (
    (1, 'None'),
    (2, 'Natural Language Explanation'),
    (3, 'Trigger Explanation')
)

#1-glove 2-w2v 3-fasttext 4-bert 5-elmo 6-gpt
EMBEDDING_SETTINGS_MAP = {
    1 : "glove",
    2 : "w2v",
    3 : "fasttext",
    4 : "bert",
    5 : "elmo",
    6 : "gpt"
}

RECOMMENDATION_SETTINGS_MAP = {
    1 : "noun_chunk",
    2 : "online_learning",
    3 : "dictionary_match"
}

EXPLANATION_SETTINGS_MAP = {
    1 : "none",
    2 : "natural_language_explanations",
    3 : "trigger_explanations"
}

ACTIVE_LEARNING_MAP = {
    0 : "none",
    1 : "random",
    2 : "mnlp"
}

EXPLANATION_SEPERATOR = ":*:*:"
