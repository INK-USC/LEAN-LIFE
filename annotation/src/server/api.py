from collections import Counter
from itertools import chain
import json
import ijson

from django.db.utils import IntegrityError
from django.shortcuts import get_object_or_404, get_list_or_404
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import viewsets, generics, filters, mixins
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework import exceptions
from rest_framework import parsers

from .models import Project, Label, Document, Setting, NamedEntityAnnotationHistory, Annotation,\
                    TriggerExplanation, NaturalLanguageExplanation, RelationExtractionAnnotationHistory
from .permissions import IsAdminUserAndWriteOnly, IsProjectUser, IsOwnAnnotation
from .serializers import ProjectSerializer, LabelSerializer, DocumentSerializer, SettingSerializer,\
                         NamedEntityAnnotationHistorySerializer, CreateBaseAnnotationSerializer,\
                         NamedEntityAnnotationSerializer, SingleUserAnnotatedDocumentSerializer,\
                         RelationExtractionAnnotationSerializer, TriggerExplanationSerializer,\
                         NaturalLanguageExplanationSerializer, SingleUserAnnotatedDocumentExplanationsSerializer,\
                         SentimentAnalysisAnnotationSerializer, RelationExtractionAnnotationHistorySerializer
from .utils import SPACY_WRAPPER
import time
from django.db import transaction

class ImportFileError(Exception):
    def __init__(self, message):
        self.message = message

class MethodSerializerView(object):
    '''
    Utility class for get different serializer class by method.
    For example:
    method_serializer_classes = {
        ('GET', ): MyModelListViewSerializer,
        ('PUT', 'PATCH'): MyModelCreateUpdateSerializer
    }
    '''
    method_serializer_options = None

    def get_serializer_class(self):
        assert self.method_serializer_options is not None, (
            'Expected view %s should contain method_serializer_options '
            'to get right serializer class.' %
            (self.__class__.__name__, )
        )
        for methods, serializer_options in self.method_serializer_options.items():
            if self.request.method in methods:
                for key in serializer_options:
                    if key in self.request.path:
                        return serializer_options[key]   

        raise exceptions.MethodNotAllowed(self.request.method)

class MethodQuerysetView(object):
    '''
    Utility class for get different queryset class by method.
    For example:
    method_serializer_classes = {
        ('GET', ): Object_One.all(),
        ('PUT', 'PATCH'): Object_Two.all()
    }
    '''
    method_queryset_options = None

    def get_queryset(self):
        assert self.method_queryset_options is not None, (
            'Expected view %s should contain method_queryset_options '
            'to get right queryset.' %
            (self.__class__.__name__, )
        )
        for methods, queryset_options in self.method_queryset_options.items():
            if self.request.method in methods:
                for key in queryset_options:
                    if key in self.request.path:
                        return queryset_options[key]   

        raise exceptions.MethodNotAllowed(self.request.method)


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    pagination_class = None
    permission_classes = (IsAuthenticated, IsAdminUserAndWriteOnly)

    def get_queryset(self):
        return self.request.user.projects

    @action(methods=['get'], detail=True)
    def progress(self, request, pk=None):
        project = self.get_object()
        return Response(project.get_progress())


class ProjectRetrieveView(generics.RetrieveAPIView):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    lookup_url_kwarg = 'project_id'


class LabelList(generics.ListCreateAPIView):
    queryset = Label.objects.all()
    serializer_class = LabelSerializer
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUserAndWriteOnly)

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs['project_id'], user_provided=False)
        return queryset

    def perform_create(self, serializer):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        serializer.save(project=project)


class ProjectStatsAPI(APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUserAndWriteOnly)

    def _get_is_annotated(self, project):
        response = project.get_progress()
        if response["annotated_doc_count"] > 0:
            return True
        return False

    def get(self, request, *args, **kwargs):
        p = get_object_or_404(Project, pk=self.kwargs['project_id'])
        
        if "heading_status" in self.request.path:
            is_annotated = self._get_is_annotated(p)
            return Response({
                "is_annotated" : is_annotated,
                "has_documents" : len(p.get_documents()) > 0
            })

        labels = [label.text for label in p.labels.all()]
        users = [user.username for user in p.users.all()]
        docs = Document.objects.get_project_docs_with_labels(self.kwargs['project_id'])
       
        nested_labels = [[a.label.text for a in doc.annotations.all()] for doc in docs]
        nested_users = [[a.user.username for a in doc.annotations.all()] for doc in docs]

        label_count = Counter(chain(*nested_labels))
        label_data = [label_count[name] for name in labels]

        user_count = Counter(chain(*nested_users))
        user_data = [user_count[name] for name in users]

        response = {'label': {'labels': labels, 'data': label_data},
                    'user': {'users': users, 'data': user_data}}

        return Response(response)


class LabelDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Label.objects.all()
    serializer_class = LabelSerializer
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUser)

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs['project_id'])

        return queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, pk=self.kwargs['label_id'])
        self.check_object_permissions(self.request, obj)

        return obj


class DocumentList(generics.ListCreateAPIView):
    queryset = Document.objects.all()
    filter_backends = (DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter)
    search_fields = ('text',)
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUserAndWriteOnly)
    serializer_class = SingleUserAnnotatedDocumentSerializer

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs['project_id'])
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        # TODO fix this, you know what
        explanation = project.explanation_type

        if explanation != 1:
            self.serializer_class = SingleUserAnnotatedDocumentExplanationsSerializer

        # TODO Remove all this logic, annotation server should be
        # able to send back correct doc ids
        if not self.request.query_params.get('active_indices'):
            return queryset
        
        
        active_indices = self.request.query_params.get('active_indices')
        active_indices = list(map(int, active_indices.split(",")))

        queryset = project.get_index_documents(active_indices)
    
        return queryset


class DocumentDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    permission_classes = (IsAuthenticated, IsProjectUser, IsAdminUser)

    def get_queryset(self):
        queryset = self.queryset.filter(project=self.kwargs['project_id'])
        return queryset

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        obj = get_object_or_404(queryset, pk=self.kwargs['doc_id'])
        self.check_object_permissions(self.request, obj)
        return obj

    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

class BaseAnnotationCreateAndDestroyView(generics.GenericAPIView, mixins.CreateModelMixin, mixins.DestroyModelMixin):
    permission_classes = (IsAuthenticated, IsProjectUser, IsOwnAnnotation)

    queryset = Annotation.objects.all()
    serializer_class = CreateBaseAnnotationSerializer

    def perform_create(self, serializer):
        doc = get_object_or_404(Document, pk=self.kwargs['doc_id'])
        task = doc.project.task
        serializer.save(document=doc, user=self.request.user, task=task)
    
    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)
    
    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)


class AnnotationDecoratorCreateView(MethodSerializerView, generics.GenericAPIView, mixins.CreateModelMixin):
    permission_classes = (IsAuthenticated, IsProjectUser, IsOwnAnnotation)
    queryset = None

    method_serializer_options = {
        ('POST') : {
            'ner' : NamedEntityAnnotationSerializer,
            're' : RelationExtractionAnnotationSerializer,
            'trigger' : TriggerExplanationSerializer,
            'nl' : NaturalLanguageExplanationSerializer,
            'sa' : SentimentAnalysisAnnotationSerializer
        }
    }

    def perform_create(self, serializer):
        annotation_id = self.kwargs['annotation_id']
        serializer.save(annotation_id=annotation_id)
    
    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)


class ExplanationDestroyView(MethodQuerysetView, generics.GenericAPIView, mixins.DestroyModelMixin):
    method_queryset_options = {
        ('DELETE') : {
            'trigger' : TriggerExplanation.objects.all(),
            'nl' : NaturalLanguageExplanation.objects.all()
        }
    }
    serializer_class = None

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)


class SettingList(generics.GenericAPIView, mixins.CreateModelMixin, mixins.UpdateModelMixin):
    queryset = Setting.objects.all()
    serializer_class = SettingSerializer
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser)

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])

        self.queryset = self.queryset.filter(project=project).filter(user__in=[project.creator, self.request.user])

        return self.queryset

    def get_object(self):
        objects = self.filter_queryset(self.get_queryset()).select_related("user").all()
        obj = None
        if len(objects) > 1:
            for settings in objects:
                if settings.user == self.request.user:
                    obj = settings
        else:
            obj = objects.get()

        self.check_object_permissions(self.request, obj)
        return obj

    def perform_create(self, serializer):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        serializer.save(project=project, user=self.request.user)

    def get(self, request, *args, **kwargs):
        return Response(self.serializer_class(self.get_object()).data)

    def put(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        _, created = Setting.objects.get_or_create(project=project, user=self.request.user,
                                                   defaults=self.request.data)
        if not created:
            return self.update(request, *args, **kwargs)
        return Response(created)


class HistoryListView(MethodSerializerView, MethodQuerysetView, generics.ListCreateAPIView):
    permission_classes = (IsAuthenticated, IsProjectUser)

    method_queryset_options = {
        ('GET') : {
            'ner' : NamedEntityAnnotationHistory.objects.all(),
            're' : RelationExtractionAnnotationHistory.objects.all()
        }
    }

    method_serializer_options = {
        ('POST', 'GET') : {
            'ner' : NamedEntityAnnotationHistorySerializer,
            're' : RelationExtractionAnnotationHistorySerializer,
        }
    }

    def perform_create(self, serializer):
        try:
            project = get_object_or_404(Project, pk=self.kwargs['project_id'])
            serializer.save(project=project, user=self.request.user)
        except IntegrityError:
            print("The word with that label is already exist in history.")
    
    def get_queryset(self):
        queryset = MethodQuerysetView.get_queryset(self)
        return queryset.filter(project=self.kwargs['project_id'])

class HistoryDestroyView(MethodQuerysetView, generics.GenericAPIView, mixins.DestroyModelMixin):
    method_queryset_options = {
        ('DELETE') : {
            'ner' : NamedEntityAnnotationHistory.objects.all(),
            're' : RelationExtractionAnnotationHistory.objects.all()
        }
    }
    serializer_class = None

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)

class AnnotationHistoryFileUpload(APIView):
    def _process_ner(self, data, all_labels, user, project, max_batch=500):
        history = []
        for entry in data:
            label_info = entry["label"].lower()
            if label_info in all_labels:
                obj = NamedEntityAnnotationHistory(user=user, project=project,\
                                                word=entry["word"], label=all_labels[label_info],\
                                                user_provided=True)
                history.append(obj)
            if len(history) == max_batch:
                NamedEntityAnnotationHistory.objects.bulk_create(history, ignore_conflicts=True)
                history = []
        
        if len(history):
            NamedEntityAnnotationHistory.objects.bulk_create(history, ignore_conflicts=True)
        return True
    
    def _process_re(self, data, all_labels, user, project, max_batch=500):
        history = []
        for entry in data:
            label_info = entry["label"].lower()
            if label_info in all_labels:
                obj = RelationExtractionAnnotationHistory(user=user, project=project,\
                                                        word_1=entry["word_1"], word_2=entry["word_2"],\
                                                        label=all_labels[label_info], user_provided=True)
                history.append(obj)
            if len(history) == max_batch:
                RelationExtractionAnnotationHistory.objects.bulk_create(history, ignore_conflicts=True)
                history = []
        
        if len(history):
            RelationExtractionAnnotationHistory.objects.bulk_create(history, ignore_conflicts=True)
        return True
    def post(self, request, *args, **kwargs):
        action = request.POST['action']
        task = request.POST['task']
        user = self.request.user
        project_id = self.kwargs["project_id"]
        project = get_object_or_404(Project, pk=project_id)
        all_labels = {}
        for label in Label.objects.all().filter(project = project_id):
            label_text = label.text.lower()
            all_labels[label_text] = label
        try:
            data_file = request.FILES['history']
            try:
                if data_file.multiple_chunks():
                    data = ijson.items(data_file, "data.item")
                else:
                    data = json.load(data_file)
                    data = data["data"]
                with transaction.atomic():
                    if action == "replace":
                        if task == "2":
                            ids = NamedEntityAnnotationHistory.objects.all()\
                                .filter(project=project_id)\
                                .filter(user_provided=True)\
                                .delete()
                        else:
                            ids = RelationExtractionAnnotationHistory.objects.all()\
                                .filter(project=project_id)\
                                .filter(user_provided=True)\
                                .delete()
                    if task == "2":
                        self._process_ner(data, all_labels, user, project)
                    else:
                        self._process_re(data, all_labels, user, project)

                    return Response(status=204)
            except:
                return Response(status=500)
        except:
            raise ImportFileError("No file was uploaded")

class RecommendationList(APIView):
    pagination_class = None
    permission_classes = (IsAuthenticated, IsProjectUser)

    def create_re_recommendations(self, doc, entity_pairs):
        entity_positions = {}
        doc_id = doc.id
        for annotation in doc.annotations.all():
            if annotation.user_provided:
                ner_position = annotation.named_entity_annotation
                key = str(ner_position.start_offset) + ":" + str(ner_position.end_offset)
                entity_positions[key] = 1
        
        recs = {}
        for pair in entity_pairs:
            if pair["sbj_offset"] in entity_positions and pair["obj_offset"] in entity_positions:
                sbj_offsets = pair["sbj_offset"].split(":")
                obj_offsets = pair["obj_offset"].split(":")
                key = pair["sbj_offset"] + "---" + pair["obj_offset"]
                rec = {
                    "doc_id" : doc_id,
                    "key" : key,
                    "label" : pair["label"]
                }
                if key in recs:
                    recs[key].append(rec)
                else:
                    recs[key] = [rec]
            
        return recs    

    def create_ner_recommendations(self, doc_id, entities):
        recommendations = {}
        for entity in entities:
            rec = {
                "document" : doc_id,
                "start_offset" : entity["start_offset"],
                "end_offset" : entity["end_offset"]
            }

            if "label" in entity:
                rec["label"] = entity["label"]
            else:
                rec["label"] = None
            
            key = range(rec["start_offset"], rec["end_offset"])
            recommendations[key] = rec
        
        return recommendations
    
    def _resolve_ner_key(self, key, current_keys):
        key_as_set = set(key)
        keys_to_delete = []
        for cur_key in current_keys:
            if len(key_as_set.intersection(cur_key)):
                keys_to_delete.append(cur_key)

        return keys_to_delete

    def resolve_ner_keys(self, keys, current_keys):
        keys_to_delete = {}
        for key in keys:
            delete_list = self._resolve_ner_key(key, current_keys)
            for d_key in delete_list:
                keys_to_delete[d_key] = 1
        
        return keys_to_delete.keys()

    def resolve_recs(self, keys_to_delete, cur_recs, new_recs):
        for key in keys_to_delete:
            del cur_recs[key]
        
        for key in new_recs:
            cur_recs[key] = new_recs[key]
        
        return cur_recs

    def get(self, request, *args, **kwargs):
        # project, document
        doc_id = self.kwargs['doc_id']
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        task_id = self.kwargs['task_id']
        document = project.documents.get(id=doc_id)

        # settings
        setting_queryset = Setting.objects.all()
        serializer_class = SettingSerializer
        setting_queryset = setting_queryset.filter(project=project, user=self.request.user)
        setting_obj = get_object_or_404(setting_queryset)
        setting_data = serializer_class(setting_obj).data
        opt_n = setting_data['noun_chunk']
        opt_m = setting_data['model_backed_recs']
        opt_h = setting_data['history']
        recommendations = {}
        cur_recommendation_keys = []

        if opt_n:
            noun_phrases = SPACY_WRAPPER.get_noun_phrases(document.text)
            recs = self.create_ner_recommendation(doc_id, noun_phrases)
            for key in recs:
                recommendations[key] = recs[key]
            
            cur_recommendation_keys = recommendations.keys()
        
        if opt_h:
            if task_id == 2:
                history_queryset = NamedEntityAnnotationHistory.objects.all()
                serializer_class = NamedEntityAnnotationHistorySerializer
                history_queryset = history_queryset.filter(project=project)
                if history_queryset.count() > 0:
                    history_obj = get_list_or_404(history_queryset)
                    h_list = serializer_class(history_obj, many=True).data
                    formatted_h_list = []
                    for entry in h_list:
                        offsets = SPACY_WRAPPER.get_char_offsets_of_noun_phrase(entry["word"], document.text)
                        for offset_pair in offsets:
                            h_dict = {
                                "label" : entry["label"],
                                "start_offset" : offset_pair["start_offset"],
                                "end_offset" : offset_pair["end_offset"]
                            }
                            formatted_h_list.append(h_dict)
    
                    recs = self.create_ner_recommendations(doc_id, formatted_h_list)

                    keys_to_delete = self.resolve_ner_keys(recs.keys(), cur_recommendation_keys)
                    recommendations = self.resolve_recs(keys_to_delete, recommendations, recs)
            elif task_id == 3:
                history_queryset = RelationExtractionAnnotationHistory.objects.all()
                serializer_class = RelationExtractionAnnotationHistorySerializer
                history_queryset = history_queryset.filter(project=project)
                if history_queryset.count() > 0:
                    history_obj = get_list_or_404(history_queryset)
                    h_list = serializer_class(history_obj, many=True).data
                    formatted_h_list = []
                    for entry in h_list:
                        sbj_offsets = SPACY_WRAPPER.get_char_offsets_of_noun_phrase(entry["word_1"], document.text)
                        obj_offsets = SPACY_WRAPPER.get_char_offsets_of_noun_phrase(entry["word_2"], document.text)
                        for i in range(len(sbj_offsets)):
                            sbj_offset = sbj_offsets[i]
                            for j in range(len(obj_offsets)):
                                obj_offset = obj_offsets[j]
                                h_dict = {
                                    "label" : entry["label"],
                                    "sbj_offset" : str(sbj_offset["start_offset"]) + ":" + str(sbj_offset["end_offset"]),
                                    "obj_offset" : str(obj_offset["start_offset"]) + ":" + str(obj_offset["end_offset"])
                                }

                                formatted_h_list.append(h_dict)
                    
                    recommendations = self.create_re_recommendations(document, formatted_h_list)
        
        if len(recommendations):
            labels = Label.objects.all().filter(project=self.kwargs['project_id'])
            label_dict = {}
            for label in labels:
                label_dict[label.id] = LabelSerializer(label).data
                        
            for key in recommendations:
                rec = recommendations[key]
                
                if task_id == 3:
                    for r in rec:
                        if r["label"] != None:
                            r["label"] = label_dict[r["label"]]
                # TODO: Make NER also handle multi recs per annotation
                else:
                    if rec["label"] != None:
                        rec["label"] = label_dict[rec["label"]]
            
        return Response({"recommendation": list(recommendations.values())})
