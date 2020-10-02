# Data Examples

## Annotation Data Examples

The data in this section can be used to create dummy projects in LEAN-LIFE to see how annotating works. We take our examples from real datasets for each task.

### conll2003_ner_example.csv

While we strongly advise people to use our [supported json data format](https://github.com/INK-USC/LEAN-LIFE/wiki/Data-Formats#ner), the import format for both Named Entity Recognition and Sentiment Analysis (Text Classification) are the same, hence we created one dummy data file using one of our supported csv formats and our SA example using our json format. We grabbed this data from the [CONLL 2003 NER benchmark dataset](https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003).

### semeval2007_re_example.json

Our RE annotation example data comes from [SemEval 2007 task 4](https://www.aclweb.org/anthology/S07-1003.pdf)

### imdb_sa_example.json

Our SA annotation example data from the the [IMDB sentiment analysis corpus](https://paperswithcode.com/sota/sentiment-analysis-on-imdb). 

## Annotation History Examples

LEAN-LIFE offers support for uploading a seed list of annotations to act as distant supervision for NER and RE tasks. If a word/phrase appears in a unlabeled text sequence we can suggest to the annotator that it is likely an entity of a certain type, similarly for relations. These two example data files can be used to test out our recommendation-via-annotation-history functionality, which can be accessed from the Settings Page. These files are to be used with the provided annotation example files above. In order to properly use these files, please make sure to create the labels stated in each file on the Label-Space-Creation Page.

### annotation_history_example_ner.json

Example of seed/historical NER annotations

### annotation_history_example_re.json

Example of seed/historical RE annotations


