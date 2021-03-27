PARSER_TRAIN_SAMPLE = 50000

DEV_F1_SAMPLE = 20000

FIND_MODULE_HIDDEN_DIM = 300

UNMATCH_TYPE_SCORE = 0

SPACY_NERS = ['', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP',
              'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

# HERE is where you can add conversions from spaCy's label space into your own
# You CANNOT pass this in, must add a constant value in here and then update
# `load_spacy_to_custom_dataset_ner_mapping()` in next_util_functions.py
SPACY_TO_TACRED = {
    'CARDINAL' : 'NUMBER',
    'GPE' : 'STATE_OR_PROVINCE',
    'LOC' : 'LOCATION',
    'ORG' : 'ORGANIZATION'
}

# HERE is where you can add your own data's NER label space, though it is better to just pass it in.
TACRED_NERS = ['PERSON', 'ORGANIZATION', 'DATE', 'NUMBER', 'TITLE', 'COUNTRY', 'LOCATION', 'CITY', 'MISC',
               'STATE_OR_PROVINCE', 'DURATION', 'NATIONALITY', 'CAUSE_OF_DEATH', 'CRIMINAL_CHARGE', 'RELIGION',
               'URL', 'IDEOLOGY']

# HERE is where you can add your own data's mapping between a relation and the entity types associated with it
# It is BETTER to just pass this datstructure in.
# It is used when providing soft-scores to a sentence:
#   if the NER types of the SUBJ and OBJ in the sentence match the explanation's label's NER types apply score, else score is 0
TACRED_ENTITY_TYPES = {
    'per:title': ('PERSON', 'TITLE'),
    'org:top_members/employees': ('ORGANIZATION', 'PERSON'),
    'per:employee_of': ('PERSON', 'ORGANIZATION'),
    'org:alternate_names': ('ORGANIZATION', 'ORGANIZATION'),
    'org:country_of_headquarters': ('ORGANIZATION', 'COUNTRY'),
    'per:countries_of_residence': ('PERSON', 'COUNTRY'),
    'org:city_of_headquarters': ('ORGANIZATION', 'CITY'),
    'per:cities_of_residence': ('PERSON', 'CITY'),
    'per:age': ('PERSON', 'NUMBER'),
    'per:stateorprovinces_of_residence': ('PERSON', 'STATE_OR_PROVINCE'),
    'per:origin': ('PERSON', 'NATIONALITY'),
    'org:subsidiaries': ('ORGANIZATION', 'ORGANIZATION'),
    'org:parents': ('ORGANIZATION', 'ORGANIZATION'),
    'per:spouse': ('PERSON', 'PERSON'),
    'org:stateorprovince_of_headquarters': ('ORGANIZATION', 'STATE_OR_PROVINCE'),
    'per:children': ('PERSON', 'PERSON'),
    'per:other_family': ('PERSON', 'PERSON'),
    'per:alternate_names': ('PERSON', 'PERSON'),
    'org:members': ('ORGANIZATION', 'ORGANIZATION'),
    'per:siblings': ('PERSON', 'PERSON'),
    'per:schools_attended': ('PERSON', 'ORGANIZATION'),
    'per:parents': ('PERSON', 'PERSON'),
    'per:date_of_death': ('PERSON', 'DATE'),
    'org:member_of': ('ORGANIZATION', 'ORGANIZATION'),
    'org:founded_by': ('ORGANIZATION', 'PERSON'),
    'org:website': ('ORGANIZATION', 'URL'),
    'per:cause_of_death': ('PERSON', 'CAUSE_OF_DEATH'),
    'org:political/religious_affiliation': ('ORGANIZATION', 'RELIGION'),
    'org:founded': ('ORGANIZATION', 'DATE'),
    'per:city_of_death': ('PERSON', 'CITY'),
    'org:shareholders': ('ORGANIZATION', 'PERSON'),
    'org:number_of_employees/members': ('ORGANIZATION', 'NUMBER'),
    'per:date_of_birth': ('PERSON', 'DATE'),
    'per:city_of_birth': ('PERSON', 'CITY'),
    'per:charges': ('PERSON', 'CRIMINAL_CHARGE'),
    'per:stateorprovince_of_death': ('PERSON', 'STATE_OR_PROVINCE'),
    'per:religion': ('PERSON', 'RELIGION'),
    'per:stateorprovince_of_birth': ('PERSON', 'STATE_OR_PROVINCE'),
    'per:country_of_birth': ('PERSON', 'COUNTRY'),
    'org:dissolved': ('ORGANIZATION', 'DATE'),
    'per:country_of_death': ('PERSON', 'COUNTRY')
 }

# USED in run_scripts ONLY, not used in pipeline code.
TACRED_LABEL_MAP = {
    'per:title': 0,
    'org:top_members/employees': 1,
    'per:employee_of': 2,
    'org:alternate_names': 3,
    'org:country_of_headquarters': 4,
    'per:countries_of_residence': 5,
    'org:city_of_headquarters': 6,
    'per:cities_of_residence': 7,
    'per:age': 8,
    'per:stateorprovinces_of_residence': 9,
    'per:origin': 10,
    'org:subsidiaries': 11,
    'org:parents': 12,
    'per:spouse': 13,
    'org:stateorprovince_of_headquarters': 14,
    'per:children': 15,
    'per:other_family': 16,
    'per:alternate_names': 17,
    'org:members': 18,
    'per:siblings': 19,
    'per:schools_attended': 20,
    'per:parents': 21,
    'per:date_of_death': 22,
    'org:member_of': 23,
    'org:founded_by': 24,
    'org:website': 25,
    'per:cause_of_death': 26,
    'org:political/religious_affiliation': 27,
    'org:founded': 28,
    'per:city_of_death': 29,
    'org:shareholders': 30,
    'org:number_of_employees/members': 31,
    'per:date_of_birth': 32,
    'per:city_of_birth': 33,
    'per:charges': 34,
    'per:stateorprovince_of_death': 35,
    'per:religion': 36,
    'per:stateorprovince_of_birth': 37,
    'per:country_of_birth': 38,
    'org:dissolved': 39,
    'per:country_of_death': 40,
    'no_relation': 41
}

# USED in run_scripts ONLY, not used in pipeline code.
TACRED_LABEL_REVERSE_MAP = {
    0: 'per:title',
    1: 'org:top_members/employees',
    2: 'per:employee_of',
    3: 'org:alternate_names',
    4: 'org:country_of_headquarters',
    5: 'per:countries_of_residence',
    6: 'org:city_of_headquarters',
    7: 'per:cities_of_residence',
    8: 'per:age',
    9: 'per:stateorprovinces_of_residence',
    10: 'per:origin',
    11: 'org:subsidiaries',
    12: 'org:parents',
    13: 'per:spouse',
    14: 'org:stateorprovince_of_headquarters',
    15: 'per:children',
    16: 'per:other_family',
    17: 'per:alternate_names',
    18: 'org:members',
    19: 'per:siblings',
    20: 'per:schools_attended',
    21: 'per:parents',
    22: 'per:date_of_death',
    23: 'org:member_of',
    24: 'org:founded_by',
    25: 'org:website',
    26: 'per:cause_of_death',
    27: 'org:political/religious_affiliation',
    28: 'org:founded',
    29: 'per:city_of_death',
    30: 'org:shareholders',
    31: 'org:number_of_employees/members',
    32: 'per:date_of_birth',
    33: 'per:city_of_birth',
    34: 'per:charges',
    35: 'per:stateorprovince_of_death',
    36: 'per:religion',
    37: 'per:stateorprovince_of_birth',
    38: 'per:country_of_birth',
    39: 'org:dissolved',
    40: 'per:country_of_death',
    41: 'no_relation'
 }