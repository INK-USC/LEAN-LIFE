# Data Examples

- We support two formats for uploading data -
  1. `JSON` (recommended) - can import documents and optionally supports metadata, annotation and explanation uploading for the document.
  2. `CSV` - can import documents, and its related metadata(optional). Doesn't support annotation and explanation uploading.

## Annotation Data Examples

The data in this section can be used to create dummy projects in LEAN-LIFE to see how annotating works. We take our examples from real datasets for each task.

### Named Entity Recognition

- We grabbed the data from the [CONLL 2003 NER benchmark dataset](https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003).
- CSV sample that just uploads text document [here](ner/conll2003_ner_none_example.csv).
- JSON sample with annotation and Trigger explanations [here](ner/conll2003_ner_none_example.csv)

### Relation Extraction

- Our RE annotation example data comes from [SemEval 2007 task 4](https://www.aclweb.org/anthology/S07-1003.pdf)
- We just support JSON format for Relation Extraction tasks.
- JSON sample with sample annotation and no explanations [here](re/semeval2007_re_none_example.json)

### Sentiment Analysis

- Our SA annotation example data from the the [IMDB sentiment analysis corpus](https://paperswithcode.com/sota/sentiment-analysis-on-imdb).
- JSON sample with annotation and no explanations [here](sa/imdb_sa_none_example.json)
- JSON sample with annotation and Natural Language explanations [here](sa/imdb_sa_nle_example.json)

## Annotation History Examples

LEAN-LIFE offers support for uploading a seed list of annotations to act as distant supervision for NER and RE tasks. If a word/phrase appears in a unlabeled text sequence we can suggest to the annotator that it is likely an entity of a certain type, similarly for relations. These two example data files can be used to test out our recommendation-via-annotation-history functionality, which can be accessed from the Settings Page. These files are to be used with the provided annotation example files above. In order to properly use these files, please make sure to create the labels stated in each file on the Label-Space-Creation Page.

### annotation_history_example_ner.json

Example of seed/historical NER annotations

### annotation_history_example_re.json

Example of seed/historical RE annotations
