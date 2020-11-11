from django.conf.urls import url
from django.urls import path
from rest_framework import routers
from rest_framework_jwt.views import obtain_jwt_token, refresh_jwt_token

from .views import LoginRedirectView
from .views import ProjectView, DatasetView, DataUpload, LabelView, StatsView, SettingView, AnnotationHistoryView
from .views import edit_form, DataDownload, DataDownloadFile
from .api import ProjectViewSet, LabelList, ProjectStatsAPI, LabelDetail, \
    BaseAnnotationCreateAndDestroyView, DocumentList, RecommendationList, DocumentDetail, \
    SettingList, AnnotationDecoratorCreateView, ExplanationDestroyView, ProjectRetrieveView, \
    HistoryListView, HistoryDestroyView, AnnotationHistoryFileUpload, UserRetrieveAPIView, TaskRetrieveAPIView, ExplanationAPIView,\
    FileUploadAPIView

router = routers.DefaultRouter()
router.register('projects', ProjectViewSet)

urlpatterns = [
    url('api/auth/obtain_token/', obtain_jwt_token),
    url('api/auth/refresh_token/', refresh_jwt_token),

    path('', LoginRedirectView.as_view(), name='index'),
    path('api/projects', ProjectRetrieveView.as_view(), name="get-all-project"),
    path('api/projects/<int:project_id>', ProjectRetrieveView.as_view(), name='project-info'),
    path('api/projects/<int:project_id>/heading_status', ProjectStatsAPI.as_view(), name='project_menu_headers'),
    path('api/projects/<int:project_id>/stats/', ProjectStatsAPI.as_view(), name='stats-api'),
    path('api/projects/<int:project_id>/settings/', SettingList.as_view(), name='settings'),
    path('api/projects/<int:project_id>/history/seed/', AnnotationHistoryFileUpload.as_view(), name='upload_seed_history'),
    path('api/projects/<int:project_id>/history/ner/', HistoryListView.as_view(), name='get_and_create_ner_history'),
    path('api/projects/<int:project_id>/history/re/', HistoryListView.as_view(), name='get_and_create_re_history'),
    path('api/projects/<int:project_id>/history/ner/<int:pk>', HistoryDestroyView.as_view(), name='delete_ner_history'),
    path('api/projects/<int:project_id>/history/re/<int:pk>', HistoryDestroyView.as_view(), name='delete_re_history'),
    path('api/projects/<int:project_id>/labels/', LabelList.as_view(), name='labels'),
    path('api/projects/<int:project_id>/labels/<int:label_id>', LabelDetail.as_view(), name='label'),
    path('api/projects/<int:project_id>/docs/', DocumentList.as_view(), name='docs'),
    path('api/projects/<int:project_id>/docs/upload/', FileUploadAPIView.as_view(), name="upload file"),
    path('api/projects/<int:project_id>/docs/<int:doc_id>', DocumentDetail.as_view(), name='doc'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/annotations/', BaseAnnotationCreateAndDestroyView.as_view(), name='create_annotation'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/annotations/<int:pk>', BaseAnnotationCreateAndDestroyView.as_view(), name='delete_annotation'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/annotations/<int:annotation_id>/ner/', AnnotationDecoratorCreateView.as_view(), name='create_ner_extension'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/annotations/<int:annotation_id>/re/', AnnotationDecoratorCreateView.as_view(), name='create_re_extension'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/annotations/<int:annotation_id>/sa/', AnnotationDecoratorCreateView.as_view(), name='create_sa_extension'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/annotations/<int:annotation_id>/nl/', AnnotationDecoratorCreateView.as_view(), name='create_nl_explanation'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/annotations/<int:annotation_id>/trigger/', AnnotationDecoratorCreateView.as_view(), name='create_trigger_explanation'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/annotations/<int:annotation_id>/nl/<int:pk>', ExplanationDestroyView.as_view(), name='delete_nl_explanation'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/annotations/<int:annotation_id>/trigger/<int:pk>', ExplanationDestroyView.as_view(), name='delete_trigger_explanation'),
    path('api/projects/<int:project_id>/docs/<int:doc_id>/recommendations/<int:task_id>/', RecommendationList.as_view(), name='recommendations'),
    path('api/users/', UserRetrieveAPIView.as_view(), name="get_users"),
    path('api/tasks/', TaskRetrieveAPIView.as_view(), name="get_tasks"),
    path('api/explanations/', ExplanationAPIView.as_view(), name="get_explanation")
    path('projects/', edit_form, name='projects'),
    path('projects/<int:project_id>/update', edit_form, name='update_projects'),
    path('projects/<int:project_id>/download', DataDownload.as_view(), name='download'),
    path('projects/<int:project_id>/download_file', DataDownloadFile.as_view(), name='download_file'),
    path('projects/<int:project_id>/', ProjectView.as_view(), name='annotation'),
    path('projects/<int:project_id>/docs/', DatasetView.as_view(), name='dataset'),
    path('projects/<int:project_id>/docs/create', DataUpload.as_view(), name='upload'),
    path('projects/<int:project_id>/labels/', LabelView.as_view(), name='label-management'),
    path('projects/<int:project_id>/stats/', StatsView.as_view(), name='stats'),
    path('projects/<int:project_id>/setting/', SettingView.as_view(), name='setting'),
    path('projects/<int:project_id>/history/', AnnotationHistoryView.as_view(), name='annotation_history'),
]