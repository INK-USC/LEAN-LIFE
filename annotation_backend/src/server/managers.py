import json
from django.db import models
from django.db.models import Prefetch
from .constants import NAMED_ENTITY_RECOGNITION_VALUE, RELATION_EXTRACTION_VALUE, SENTIMENT_ANALYSIS_VALUE
from .utils import convert_task_name_to_annotation_type, convert_explanation_int_to_explanation_type, SPACY_WRAPPER


class AnnotationManager(models.Manager):
    def get_annotations_for_export(self, user_id, task_name, explanation_int):
        queryset = self.get_queryset()
        annotation_type = convert_task_name_to_annotation_type(task_name)
        if explanation_int > 1:
            explanation_type = convert_explanation_int_to_explanation_type(explanation_int)
            annotation_queryset = queryset.filter(user_id=user_id) \
                                          .select_related(annotation_type, "label") \
                                          .prefetch_related(explanation_type)
        else:
            annotation_queryset = queryset.filter(user_id=user_id) \
                                          .select_related(annotation_type, "label")
        
        return annotation_queryset


class DocumentManager(models.Manager):
    def _format_json(self, project_docs, user_id, explanation, task_name):
        dataset = {"data" : []}
        for doc in project_docs:
            annotations = []
            for a in doc.user_annotations:
                if not a.user_provided:
                    output = {

                        "annotation_id" : a.id,
                        "label" : a.label.text,
                    }

                    extended_output = a.format_json_outputs(task_name)

                    for key in extended_output:
                        output[key] = extended_output[key]

                    if explanation:
                        output["explanation"] = a.format_explanations("json", doc.text)

                    annotations.append(output)

            dataset["data"].append({
                "doc_id" : doc.id,
                "text" : doc.text,
                "annotations" : annotations,
                "user" : user_id,
                "metadata" : json.loads(doc.metadata)
            })
        return dataset
    

    def _format_csv_ner(self, project_docs, explanation):
        dataset = []

        header = ["document_id", "word", "label", "metadata"]
        if explanation:
            header.append("explanation")
        dataset.append(header)

        for i, doc in enumerate(project_docs):
            text = doc.text
            words = SPACY_WRAPPER.tokenize(text)
            if explanation:
                doc_rep = [[i+1, word.encode('utf-8'), 'O', "", ""] for word in words]
            else:
                doc_rep = [[i+1, word.encode('utf-8'), 'O', ""] for word in words]

            doc_rep[0][3] = doc.metadata

            startoff_map = {}
            start_off = 0
            for word_index, tup in enumerate(doc_rep):
                startoff_map[start_off] = word_index
                start_off = start_off + len(tup[1]) + 1

            for a in doc.user_annotations:
                start_offset = a.named_entity_annotation.start_offset
                end_offset = a.named_entity_annotation.end_offset
                if start_offset in startoff_map:
                    doc_rep[startoff_map[start_offset]][2] = 'B-{}'.format(a.label.text)
                    if explanation:
                        explanations = a.format_explanations("csv", doc.text)
                        doc_rep[startoff_map[start_offset]][4] = explanations
                    for i in range(start_offset+1, end_offset):
                        if i in startoff_map:
                            doc_rep[startoff_map[i]][2] = 'I-{}'.format(a.label.text)
            
            for row in doc_rep:
                dataset.append(row)
            dataset.append("")
            
        dataset.pop()
        return dataset

    def _format_csv_re(self, project_docs, explanation):
        dataset = []
        header = ["document_id", "entity_1", "entity_2", "label", "metadata"]
        if explanation:
            header.append("explanation")
        dataset.append(header)

        for i, doc in enumerate(project_docs):
            text = doc.text
            for j, a in enumerate(doc.user_annotations):
                if not a.user_provided:
                    tmp_start_offset = a.relation_extraction_annotation.sbj_start_offset
                    tmp_end_offset = a.relation_extraction_annotation.sbj_end_offset
                    sbj_entity = text[tmp_start_offset:tmp_end_offset]
                    
                    tmp_start_offset = a.relation_extraction_annotation.obj_start_offset
                    tmp_end_offset = a.relation_extraction_annotation.obj_end_offset
                    obj_entity = text[tmp_start_offset:tmp_end_offset]

                    metadata = doc.metadata if j == 0 else ""
                    
                    if explanation:
                        explanations = a.format_explanations("csv", doc.text)
                        dataset.append([i+1, sbj_entity.encode('utf-8'), obj_entity.encode('utf-8'), a.label.text, metadata, explanations])
                    else:
                        dataset.append([i+1, sbj_entity.encode('utf-8'), obj_entity.encode('utf-8'), a.label.text, metadata])
            
            dataset.append("")
        
        dataset.pop()
        return dataset

    def _format_csv_sa(self, project_docs, explanation):
        dataset = []
        header = ["document_id", "text", "label", "metadata"]
        if explanation:
            header.append("explanation")
        dataset.append(header)
        for i, doc in enumerate(project_docs):
            for j, a in enumerate(doc.user_annotations):
                metadata = doc.metadata if j ==0 else ""
                if explanation:
                    explanations = a.format_explanations("csv", doc.text)
                    dataset.append([i+1, doc.text.encode('utf-8'), a.label.text, metadata, explanations])
                else:
                    dataset.append([i+1, doc.text.encode('utf-8'), a.label.text, metadata])

        
        return dataset

    def export_ner_project_user_documents(self, project_id, user_id, export_format, annotation_queryset, explanation=False):
        queryset = self.get_queryset()
        # https://docs.djangoproject.com/en/3.1/ref/models/querysets/#django.db.models.Prefetch
        project_docs = queryset.filter(project_id=project_id).prefetch_related(Prefetch(
            lookup="annotations",
            queryset=annotation_queryset,
            to_attr="user_annotations"
        ))
        if export_format == "csv":
            return self._format_csv_ner(project_docs, explanation)

            
        if export_format == "json":
            return self._format_json(project_docs, user_id, explanation, NAMED_ENTITY_RECOGNITION_VALUE)
                
    
    def export_re_project_user_documents(self, project_id, user_id, export_format, annotation_queryset, explanation=False):
        queryset = self.get_queryset()

        # https://docs.djangoproject.com/en/3.1/ref/models/querysets/#django.db.models.Prefetch
        project_docs = queryset.filter(project_id=project_id).prefetch_related(Prefetch(
            lookup="annotations",
            queryset=annotation_queryset,
            to_attr="user_annotations"
        ))

        if export_format == "csv":
            return self._format_csv_re(project_docs, explanation)
        
        if export_format == "json":
            return self._format_json(project_docs, user_id, explanation, RELATION_EXTRACTION_VALUE)
    
    def export_sa_project_user_documents(self, project_id, user_id, export_format, annotation_queryset, explanation=False):
        queryset = self.get_queryset()
        project_docs = queryset.filter(project_id=project_id).prefetch_related(Prefetch(
            lookup="annotations",
            queryset=annotation_queryset,
            to_attr="user_annotations"
        ))

        if export_format == "csv":
            return self._format_csv_sa(project_docs, explanation)
    
        if export_format == "json":
           return self._format_json(project_docs, user_id, explanation, SENTIMENT_ANALYSIS_VALUE)
    
    def export_project_user_documents(self, task_name, project_id, user_id, export_format, annotation_queryset, explanation_int):
        explanation_bool = explanation_int > 1
        if task_name == NAMED_ENTITY_RECOGNITION_VALUE:
            return self.export_ner_project_user_documents(project_id, user_id, export_format, annotation_queryset, explanation_bool)
        elif task_name == RELATION_EXTRACTION_VALUE:
            return self.export_re_project_user_documents(project_id, user_id, export_format, annotation_queryset, explanation_bool)
        elif task_name == SENTIMENT_ANALYSIS_VALUE:
            return self.export_sa_project_user_documents(project_id, user_id, export_format, annotation_queryset, explanation_bool)
    
    def get_project_docs_with_labels(self, project_id):
        queryset = self.get_queryset()
        project_docs = queryset.filter(project_id=project_id) \
                               .prefetch_related("annotations__label", 
                                                 "annotations__user")
        return project_docs
