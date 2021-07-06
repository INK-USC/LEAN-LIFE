This file contains history of release notes.

# v1.0 (Date: 6/15/21)

- This release moves away from django templates to a standalone Vue.js based UI. The django templates based UI will be removed in future releases.
- The UI adds support for annotating documents, training and downloading models on the annotated documents.

- FastAPI layer is introduced to support asynchronous and standalone model training endpoints.
- Support for training of Relation Extraction and Sentiment Analysis tasks with natural language explanations using NeXT framework is added.
- It also supports standard training of RE and SA tasks when explanations types is None or trigger.

# v0.1-alpha (Date: 9/28/20)

This release focuses on providing a Web-UI to annotate sets of documents. As mentioned the user can create annotations for three tasks (NER, SA, RE), as well as provide explanations for those annotations in two different ways. We support various import/export formats, and take advantage of two recommendation strategies for NER (Noun phrase chunking--powered by spaCy, Historical Annotation Application) and one for RE (Historical Annotation Application). To further enable Historical Annotation Application, a project owner can upload a set of (x,y) pairs as "Historical" annotations, and we will apply these pairs as recommendations for un-annotated documents (sort of like distant learning).
