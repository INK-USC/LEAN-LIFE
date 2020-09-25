import json
from django.core.exceptions import ValidationError, FieldError
from django.core import validators
from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User
from django.contrib.staticfiles.storage import staticfiles_storage
from .utils import get_key_choices, convert_task_name_to_annotation_type
from .managers import DocumentManager, AnnotationManager

from .constants import TASK_CHOICES, SENTIMENT_ANALYSIS_VALUE, RELATION_EXTRACTION_VALUE, NAMED_ENTITY_RECOGNITION_VALUE,\
                       EXPLANATION_SEPERATOR, EXPLANATION_CHOICES

class Task(models.Model):
    name = models.CharField(max_length=30, choices=TASK_CHOICES)

    def get_projects(self):
        return self.projects.all()
    
    def __str__(self): 
        return self.name
    
# TODO: need to create User object that extends AUTH_USER... just to build functions for the user
# like grab dataset generated by user

class Project(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=500)
    guideline = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    creator = models.ForeignKey(on_delete=models.DO_NOTHING, related_name='created_projects', to=User)
    users = models.ManyToManyField(User, related_name='projects')
    task = models.ForeignKey(on_delete=models.DO_NOTHING, related_name='projects', to=Task)
    explanation_type = models.IntegerField(default=1, choices=EXPLANATION_CHOICES)
    
    # TODO: Seemingly not used
    def get_absolute_url(self):
        return reverse('upload', args=[self.id])
    
    def get_documents(self):
        return self.documents.all()

    # # don't love this, the need to update annotated, but I conceed here
    def get_annotated_documents(self):
        docs = self.documents.all()
        docs = docs.filter(annotated=True)
        return docs

    def get_progress(self):
        docs = self.get_annotated_documents()
        return {"annotated_doc_count" : docs.count()}

    # TOOD: Figure out if its okay to get rid of this
    @property
    def image(self):
        url = staticfiles_storage.url('images/cat-3449999_640.jpg')
        return url
    
    # TODO: Should be removed, no need for it
    def get_index_documents(self, indices):
        docs = self.documents.all()
        docs_indices = [d.id for d in docs]
        active_indices = [docs_indices[i-1] for i in indices]
        docs = list(docs.filter(pk__in=active_indices))
        docs.sort(key=lambda t: active_indices.index(t.pk))
        return docs

    def get_task_name(self):
        return self.task.name
    
    def get_document_serializer(self):
        from .serializers import UserAnnotatedDocumentSerializer
        return UserAnnotatedDocumentSerializer
    
    def get_annotation_serializer(self):
        from .serializers import AnnotationSerializer
        return AnnotationSerializer
    
    def get_dataset_for_user(self, user_id, explanation, explanation_type):
        if self.task.name == NAMED_ENTITY_RECOGNITION_VALUE:
            return Document.objects.export_ner_project_user_documents(self.id, user_id, explanation, explanation_type)
    
    # TODO: add ability to get dataset by project
    # def get_data_set(self):
    #     Annotation.objects.raw()

    def __str__(self):
        return self.name

class Label(models.Model):
    KEY_CHOICES = get_key_choices()
    COLOR_CHOICES = ()

    text = models.CharField(max_length=30)
    shortcut = models.CharField(max_length=15, blank=True, null=True, choices=KEY_CHOICES)
    project = models.ForeignKey(Project, related_name='labels', on_delete=models.CASCADE)
    background_color = models.CharField(max_length=7, default='#209cee')
    text_color = models.CharField(max_length=7, default='#ffffff')
    user_provided = models.BooleanField(default=False)

    def __str__(self):
        return self.text

    class Meta:
        unique_together = (
            ('project', 'text'),
            ('project', 'shortcut')
        )


class Document(models.Model):
    text = models.TextField()
    project = models.ForeignKey(Project, related_name='documents', on_delete=models.CASCADE)
    annotated = models.BooleanField(default=False)
    metadata = models.TextField(default='{}')
    objects = DocumentManager()
        
    def delete_annotations(self):
        self.annotations.all().delete()
    
    def delete_extended_annotations(self):
        annotations = self.get_base_annotations().select_related(self.get_annotation_type())
        annotations.delete()

    def __str__(self):
        return self.text[:50]
        
    
    class Meta:
        indexes= [
            models.Index(fields=["project"], name="document_project_index")
        ]

class Annotation(models.Model):
    prob = models.FloatField(default=0.0)
    via_recommendation = models.BooleanField(default=False)
    user_provided = models.BooleanField(default=False)
    task = models.ForeignKey(on_delete=models.CASCADE, related_name='annotations', to=Task)
    document = models.ForeignKey(on_delete=models.CASCADE, related_name='annotations', to=Document)
    user = models.ForeignKey(on_delete=models.DO_NOTHING, related_name='annotations', to=User)
    label = models.ForeignKey(on_delete=models.CASCADE, related_name='annotations', to=Label)
    objects = AnnotationManager()
    
    def get_extended_annotation(self):
        if self.task.name == NAMED_ENTITY_RECOGNITION_VALUE:
            return self.named_entity_annotation
        if self.task.name == RELATION_EXTRACTION_VALUE:
            if self.user_provided:
                return self.named_entity_annotation
            else:
                return self.relation_extraction_annotation
        if self.task.name == SENTIMENT_ANALYSIS_VALUE:
            return self.sentiment_analysis_annotation
    
    def get_extended_annotation_serializer(self):
        if self.task.name == NAMED_ENTITY_RECOGNITION_VALUE:
            from .serializers import NamedEntityAnnotationSerializer
            return NamedEntityAnnotationSerializer
        if self.task.name == RELATION_EXTRACTION_VALUE:
            if self.user_provided:
                from .serializers import NamedEntityAnnotationSerializer
                return NamedEntityAnnotationSerializer
            else:
                from .serializers import RelationExtractionAnnotationSerializer
                return RelationExtractionAnnotationSerializer
        if self.task.name == SENTIMENT_ANALYSIS_VALUE:
            from .serializers import SentimentAnalysisAnnotationSerializer
            return SentimentAnalysisAnnotationSerializer

    def _get_trigger_explanations(self):
        if self.trigger_explanations.count() < 1:
            raise FieldError("No Triggers")
        return self.trigger_explanations.all()
 
    def _get_natural_language_explantions(self):
        if self.natural_language_explanations.count() < 1:
            raise FieldError("No NL Explanations")
        return self.natural_language_explanations.all()
    
    def get_explanations(self, format=False):
        if self.trigger_explanations.count() > 0:
            if format:
                return self._get_formatted_trigger_explanations()
            else:
                return self._get_trigger_explanations()
        elif self.natural_language_explanations.count() > 0:
            if format:
                return self._get_format_natural_language_explanations()
            else:
                return self._get_natural_language_explantions()
        else:
            return None
    
    def get_explanation_serializer(self):
        if self.trigger_explanations.count() > 0:
            from .serializers import TriggerExplanationSerializer
            return TriggerExplanationSerializer
        if self.natural_language_explanations.count() > 0:
            from .serializers import NaturalLanguageExplanationSerializer
            return NaturalLanguageExplanationSerializer

    def format_explanations(self, output_format, text=None):
        if self.trigger_explanations.count() > 0:
            return TriggerExplanation.format_explanations(self.trigger_explanations, text, output_format)
        if self.natural_language_explanations.count() > 0:
            return NaturalLanguageExplanation.format_explanations(self.natural_language_explanations, output_format)
    
    # passing task_name in here, as I don't want another db call
    def format_json_outputs(self, task_name):
        if task_name == NAMED_ENTITY_RECOGNITION_VALUE:
            return self.format_ner_json_outputs()
        elif task_name == RELATION_EXTRACTION_VALUE:
            return self.format_re_json_outputs()
        elif task_name == SENTIMENT_ANALYSIS_VALUE:
            return self.format_sa_json_outputs()
    
    def format_sa_json_outputs(self):
        output = {
            # "id" : self.sentiment_analysis_annotation.id,
            "annotation_id" : self.sentiment_analysis_annotation.annotation_id
        }
        return output
    
    def format_ner_json_outputs(self):
        output = {
            "start_offset" : self.named_entity_annotation.start_offset,
            "end_offset" : self.named_entity_annotation.end_offset
        }

        return output
    
    def format_re_json_outputs(self):
        output = {
            "sbj_start_offset" : self.relation_extraction_annotation.sbj_start_offset,
            "sbj_end_offset" : self.relation_extraction_annotation.sbj_end_offset,
            "obj_start_offset" : self.relation_extraction_annotation.obj_start_offset,
            "obj_end_offset" : self.relation_extraction_annotation.obj_end_offset
        }

        return output

    def clean(self):
        if self.document.project.id != self.label.project.id:
            raise ValidationError("label and document don't come from same project")

        # TODO: if user is not assigned to project throw error -- manager

    class Meta:
        indexes = [
            # TODO: add ability to create dataset for user by project
            models.Index(fields=['user'], name='annotation_user_index'),
            # TODO: add ability to download labels by label
            models.Index(fields=['document', 'user', 'label'], name='annotation_retrieval_index'),
        ]

class NamedEntityAnnotation(models.Model):
    annotation = models.OneToOneField(to=Annotation, on_delete=models.CASCADE, related_name='named_entity_annotation')
    start_offset = models.PositiveIntegerField()
    end_offset = models.PositiveIntegerField()

    # TODO: need to make sure start and end are less than length
    def clean(self):
        if self.start_offset >= self.end_offset:
            raise ValidationError('start_offset is after end_offset')

    class Meta:
        # TODO: will have to be a function we create to check this, can remove label
        # unique_together = ('document', 'user', 'label', 'start_offset', 'end_offset', 'user_provided')
        indexes = [
            models.Index(fields=['annotation'], name='named_entity_annotation_index')
        ]

class RelationExtractionAnnotation(models.Model):
    annotation = models.OneToOneField(to=Annotation, on_delete=models.CASCADE, related_name='relation_extraction_annotation')
    sbj_start_offset = models.PositiveIntegerField()
    sbj_end_offset = models.PositiveIntegerField()
    obj_start_offset = models.PositiveIntegerField()
    obj_end_offset = models.PositiveIntegerField()

    def clean(self):
        if self.sbj_start_offset >= self.sbj_end_offset:
            raise ValidationError("sbj_start_offset is after sbj_end_offset")
        
        if self.obj_start_offset >= self.obj_end_offset:
            raise ValidationError("obj_start_offset is after obj_end_offset")

        if self.obj_start_offset <= self.sbj_end_offset and self.obj_start_offset >= self.sbj_start_offset:
            raise ValidationError("Object starts in the middle of the Subject")

        if self.sbj_start_offset <= self.obj_end_offset and self.sbj_start_offset >= self.obj_start_offset:
            raise ValidationError("Subject starts in the middle of the Object")
    
    class Meta:
        # TODO: will have to be a function we create to check this
        # unique_together = ('document', 'user', 'label', 'start_offset_1', 'end_offset_1', 'start_offset_2', "end_offset_2")
        indexes = [
            models.Index(fields=['annotation'], name='re_annotation_index')
        ]

class SentimentAnalysisAnnotation(models.Model):
    annotation = models.OneToOneField(to=Annotation, on_delete=models.CASCADE, related_name='sentiment_analysis_annotation', primary_key=True)
    

# TODO: phrases per trigger must be sorted by start_offset before insertion
class TriggerExplanation(models.Model):
    annotation = models.ForeignKey(to=Annotation, on_delete=models.CASCADE, related_name='trigger_explanations')
    trigger_id = models.PositiveIntegerField()
    start_offset = models.PositiveIntegerField()
    end_offset = models.PositiveIntegerField()

    def clean(self):
        if self.start_offset >= self.end_offset:
            raise ValidationError('start_offset is after end_offset')
    
    @staticmethod
    def format_explanations(triggers, text, output_format):
        trigger_dict = {}
        for trigger in triggers.all():
            phrase = text[trigger.start_offset:trigger.end_offset]
            if trigger.trigger_id in trigger_dict:
                trigger_dict[trigger.trigger_id] = trigger_dict[trigger.trigger_id] + " " + phrase
            else:
                trigger_dict[trigger.trigger_id] = phrase
        
        if output_format == "csv":
            return EXPLANATION_SEPERATOR.join(list(trigger_dict.values()))
        elif output_format == "json":
            return list(trigger_dict.values())

    class Meta:
        indexes = [
            models.Index(fields=['annotation', 'trigger_id'], name='trigger_annotation_index')
        ]
        unique_together = ('annotation', 'trigger_id', 'start_offset', 'end_offset')


class NaturalLanguageExplanation(models.Model):
    annotation = models.ForeignKey(to=Annotation, on_delete=models.CASCADE, related_name='natural_language_explanations')
    text = models.TextField(max_length=500, validators=[validators.MinLengthValidator(1)])

    @staticmethod
    def format_explanations(explanations, output_format):
        if output_format == "csv":
            return EXPLANATION_SEPERATOR.join([e.text for e in explanations.all()])
        elif output_format == "json":
            return [e.text for e in explanations.all()]

    class Meta:
        indexes = [
            models.Index(fields=['annotation'], name='nl_annotation_index')
        ]
        unique_together = ('annotation', 'text')


# TODO: need to write a clean function here
# check if label comes from project
class NamedEntityAnnotationHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    annotation = models.ForeignKey(to=Annotation, on_delete=models.CASCADE, related_name='ner_history', null=True)
    project = models.ForeignKey(Project, related_name='named_entity_history', on_delete=models.CASCADE)
    word = models.TextField(max_length=100)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    user_provided = models.BooleanField(default=False)

    class Meta:
        indexes = [
            models.Index(fields=['project', 'user'], name="ne_history_lookup_index")
        ]
        unique_together = ('user', 'word', 'label')


# TODO: need to write a match function here
# check if label comes from project
class RelationExtractionAnnotationHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    project = models.ForeignKey(Project, related_name='relation_extraction_history', on_delete=models.CASCADE)
    annotation = models.ForeignKey(to=Annotation, on_delete=models.CASCADE, related_name='re_history', null=True)
    word_1 = models.TextField(max_length=100)
    word_2 = models.TextField(max_length=100)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    user_provided = models.BooleanField(default=False)

    class Meta:
        indexes = [
            models.Index(fields=['project', 'user'], name="re_history_lookup_index")
        ]
        unique_together = ('user', 'word_1', 'word_2', 'label')


# TODO: Should all these settings really be attached to the user, or really be set by project lead?
class Setting(models.Model):
    user = models.ForeignKey(User, related_name='settings', on_delete=models.CASCADE)
    project = models.ForeignKey(Project, related_name='settings', on_delete=models.CASCADE)
    noun_chunk = models.BooleanField(default=False)
    model_backed_recs = models.BooleanField(default=False)
    history = models.BooleanField(default=False)
    is_online_on = models.BooleanField(default=False)
    encoding_strategy = models.IntegerField(default=1) #1-glove 2-w2v 3-fasttext 4-bert 5-elmo 6-gpt
    batch = models.IntegerField(default=10)
    epoch = models.IntegerField(default=5)
    active_learning_strategy = models.IntegerField(default=1)
    acquire = models.IntegerField(default=5)
    

    class Meta:
        indexes = [
            models.Index(fields=['project', 'user'], name="user_project_settings_index")
        ]

        unique_together = ('user', 'project')