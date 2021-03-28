from rest_framework import serializers
from django.contrib.auth.models import User

from .models import NamedEntityAnnotation, RelationExtractionAnnotation, SentimentAnalysisAnnotation
from .models import RelationExtractionAnnotationHistory, NamedEntityAnnotationHistory
from .models import Task, Label, Project, Document, Setting, Annotation
from .models import TriggerExplanation, NaturalLanguageExplanation
from .constants import NAMED_ENTITY_RECOGNITION_VALUE, RELATION_EXTRACTION_VALUE, SENTIMENT_ANALYSIS_VALUE

class ProjectFilteredPrimaryKeyRelatedField(serializers.PrimaryKeyRelatedField):

    def get_queryset(self):
        view = self.context.get('view', None)
        request = self.context.get('request', None)
        queryset = super(ProjectFilteredPrimaryKeyRelatedField, self).get_queryset()
        if not request or not queryset or not view:
            return None
        return queryset.filter(project=view.kwargs['project_id'])


class TaskSerializer(serializers.ModelSerializer):

    class Meta:
        model = Task
        fields = ('id', 'name')


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ('id', 'name', 'description', 'guideline', 'users', 'task', 'image', 'updated_at', 'explanation_type')
        extra_kwargs = {'users': {'required': False}}


class LabelSerializer(serializers.ModelSerializer):

    class Meta:
        model = Label
        fields = ('id', 'text', 'shortcut', 'background_color', 'text_color')


class DocumentSerializer(serializers.ModelSerializer):
    def __init__(self, *args, **kwargs):
        many = kwargs.pop('many', True)
        super(DocumentSerializer, self).__init__(many=many, *args, **kwargs)

    class Meta:
        model = Document
        fields = ('id', 'text', 'project', 'annotated', 'metadata')

    def update(self, instance, validated_data):
        instance.annotated = validated_data.get('annotated', instance.annotated)
        instance.save()
        return instance

    def create(self, validated_data):
        docs = [Document(**item) for item in validated_data]
        return Document.objects.bulk_create(docs)


class CreateBaseAnnotationSerializer(serializers.ModelSerializer):

    def create(self, validated_data):
        annotation = Annotation.objects.create(**validated_data)
        return annotation

    class Meta:
        model = Annotation
        fields = ('id', 'via_recommendation', 'label')


class AnnotationWithExtensionSerializer(serializers.ModelSerializer):
    label = ProjectFilteredPrimaryKeyRelatedField(queryset=Label.objects.all())
    extended_annotation = serializers.SerializerMethodField()

    def get_extended_annotation(self, instance):
        request = self.context.get('request')
        if request:
            extension = instance.get_extended_annotation()
            serializer = instance.get_extended_annotation_serializer()
            return serializer(extension).data

    class Meta:
        model = Annotation
        fields = ('id', 'label', 'extended_annotation', 'user_provided')


class AnnotationWithExplanationsSerializer(serializers.ModelSerializer):
    label = ProjectFilteredPrimaryKeyRelatedField(queryset=Label.objects.all())
    extended_annotation = serializers.SerializerMethodField()
    explanations = serializers.SerializerMethodField()

    def get_extended_annotation(self, instance):
        request = self.context.get('request')
        if request:
            extension = instance.get_extended_annotation()
            serializer = instance.get_extended_annotation_serializer()
            return serializer(extension).data

    def get_explanations(self, instance):
        request = self.context.get('request')
        if request:
            explanations = instance.get_explanations()
            if explanations:
                serializer = instance.get_explanation_serializer()
                return serializer(explanations, many=True).data
        return None

    class Meta:
        model = Annotation
        fields = ('id', 'label', 'extended_annotation', 'explanations', 'user_provided')

class SingleUserAnnotatedDocumentSerializer(serializers.ModelSerializer):
    annotations = AnnotationWithExtensionSerializer(many=True, read_only=True)

    class Meta:
        model = Document
        fields = ('id', 'text', 'annotations', 'annotated')

class SingleUserAnnotatedDocumentExplanationsSerializer(serializers.ModelSerializer):
    annotations = AnnotationWithExplanationsSerializer(many=True, read_only=True)

    class Meta:
        model = Document
        fields = ('id', 'text', 'annotations', 'annotated')

class SentimentAnalysisAnnotationSerializer(serializers.ModelSerializer):

    class Meta:
        model = SentimentAnalysisAnnotation
        fields = ('annotation_id',)

    def create(self, validated_data):
        return SentimentAnalysisAnnotation.objects.create(**validated_data)


class NamedEntityAnnotationSerializer(serializers.ModelSerializer):

    class Meta:
        model = NamedEntityAnnotation
        fields = ('id', 'start_offset', 'end_offset')

    def create(self, validated_data):
        return NamedEntityAnnotation.objects.create(**validated_data)


class RelationExtractionAnnotationSerializer(serializers.ModelSerializer):
    class Meta:
        model = RelationExtractionAnnotation
        fields = ('id', 'sbj_start_offset', 'sbj_end_offset', 'obj_start_offset', 'obj_end_offset')

    def create(self, validated_data):
        return RelationExtractionAnnotation.objects.create(**validated_data)


class TriggerExplanationSerializer(serializers.ModelSerializer):

    class Meta:
        model = TriggerExplanation
        fields = ('id', 'trigger_id', 'start_offset', 'end_offset')

    def create(self, validated_data):
        return TriggerExplanation.objects.create(**validated_data)


class NaturalLanguageExplanationSerializer(serializers.ModelSerializer):

    class Meta:
        model = NaturalLanguageExplanation
        fields = ('id', 'text')

    def create(self, validated_data):
        return NaturalLanguageExplanation.objects.create(**validated_data)


class SettingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Setting
        fields = ('id', 'noun_chunk', 'model_backed_recs', 'history', 'is_online_on', 'encoding_strategy', 'batch', 'epoch', 'active_learning_strategy', 'acquire')

    def update(self, instance, validated_data):
        instance.noun_chunk = validated_data.get('noun_chunk', instance.noun_chunk)
        instance.model_backed_recs = validated_data.get('model_backed_recs', instance.model_backed_recs)
        instance.history = validated_data.get('history', instance.history)
        instance.is_online_on = validated_data.get('is_online_on', instance.is_online_on)
        instance.encoding_strategy = validated_data.get('encoding_strategy', instance.encoding_strategy)
        instance.batch = validated_data.get('batch', instance.batch)
        instance.epoch = validated_data.get('epoch', instance.epoch)
        instance.active_learning_strategy = validated_data.get('active_learning_strategy', instance.active_learning_strategy)
        instance.acquire = validated_data.get('acquire', instance.acquire)
        instance.save()
        return instance

class NamedEntityAnnotationHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = NamedEntityAnnotationHistory
        fields = ('id', 'word', 'label', 'annotation', 'user_provided')

    def create(self, validated_data):
        return NamedEntityAnnotationHistory.objects.create(**validated_data)

class RelationExtractionAnnotationHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = RelationExtractionAnnotationHistory
        fields = ('id', 'word_1', 'word_2', 'label', 'annotation', 'user_provided')

    def create(self, validated_data):
        return RelationExtractionAnnotationHistory.objects.create(**validated_data)

from django.contrib.auth.models import User


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('username', 'id')