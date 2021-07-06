no_anchor_word_templates = [
    "The word '____' appears in the text",
    "The phrase '____' appears in the text"
]

# REPLACE is automatically replaced in the prompt by either a choice of anchor words if 2 are present, 
# or by a single anchor word if only one is present
one_anchor_word_templates = [
    "The word  '____'  appears to the right of 'REPLACE' by at most  ____  words",
    "The phrase  '____'  appears to the right of 'REPLACE' by at most  ____  words",
    "The word  '____'  appears to the left of 'REPLACE' by at most  ____  words",
    "The phrase  '____'  appears to the left of 'REPLACE' by at most  ____  words",
    "The word  '____'  appears to the right of 'REPLACE' by at least  ____  words",
    "The phrase  '____'  appears to the right of 'REPLACE' by at least  ____  words",
    "The word  '____'  appears to the left of 'REPLACE' by at least  ____  words",
    "The phrase  '____'  appears to the left of 'REPLACE' by at least  ____  words",
    "The word  '____'  appears directly to the right of 'REPLACE'",
    "The phrase  '____'  appears directly to the right of 'REPLACE'",
    "The word  '____'  appears directly to the left of 'REPLACE'",
    "The phrase  '____'  appears directly to the left of 'REPLACE'",
    "The word  '____'  appears within  ____  words of 'REPLACE'",
    "The phrase  '____'  appears within  ____  words of 'REPLACE'",
    "The word  '____'  appears within 1 word of 'REPLACE'",
    "The phrase  '____'  appears within 1 word of 'REPLACE'"
]

# Similar to above, REPLACE-1 and REPLACE-2 are both replaced by a choice of the two anchor words
two_anchor_word_templates = [
    "The phrase  '____'  appears between 'REPLACE-1' and 'REPLACE-2'",
    "The word  '____'  appears between 'REPLACE-1' and 'REPLACE-2'",
    "The word  '____'  is the only word between 'REPLACE-1' and 'REPLACE-2'",
    "There are no more than  ____   words between 'REPLACE-1' and 'REPLACE-2'",
    "There are no less than  ____   words between 'REPLACE-1' and 'REPLACE-2'",
    "There is one word between 'REPLACE-1' and 'REPLACE-2'",
    "'REPLACE-1' comes before 'REPLACE-2' by at most  ____  words",
    "'REPLACE-1' comes before 'REPLACE-2' by at least  ____  words",
    "'REPLACE-1' comes directly before 'REPLACE-2'",
]