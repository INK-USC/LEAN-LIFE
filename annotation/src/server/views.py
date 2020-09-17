import csv
import json
import ijson
from io import TextIOWrapper
import logging
from django import forms

from django.contrib.auth.views import LoginView as BaseLoginView
from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseForbidden
from django.shortcuts import get_object_or_404, render, render_to_response
from django.views import View
from django.views.generic import TemplateView, CreateView, UpdateView, RedirectView
from django.views.generic.list import ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .permissions import SuperUserMixin
from .models import Document, Project, NamedEntityAnnotationHistory, RelationExtractionAnnotationHistory, Annotation, Setting, NamedEntityAnnotation, Label
from .constants import NAMED_ENTITY_RECOGNITION_VALUE, RELATION_EXTRACTION_VALUE
from django.db import transaction

logger = logging.getLogger(__name__)

class LoginRedirectView(RedirectView):
    pattern_name = 'redirect-to-login'
    def get_redirect_url(self, *args, **kwargs):
        return '/login'


class ProjectView(LoginRequiredMixin, TemplateView):
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        context['bundle_name'] = "_".join(project.get_task_name().split(" ")).lower()
        return context

    def get_template_names(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        task_name = project.task.name
        if task_name == 'Sentiment Analysis':
            return ['annotation/sentiment_analysis.html']
        elif task_name == 'Named Entity Recognition':
            return ['annotation/ner.html']
        elif task_name == 'Relation Extraction':
            return ['annotation/relation_extraction.html']


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ('name', 'description', 'task', 'explanation_type', 'users')

@login_required
def edit_form(request, project_id=None, template_name='projects.html'):
    if project_id:
        project = get_object_or_404(Project, pk=project_id)
        if project.creator != request.user:
            return HttpResponseForbidden()
        form = ProjectForm(request.POST or None, instance=project)
    else:
        form = ProjectForm(request.POST or None)
    
    if request.POST and form.is_valid():
        if project_id:
            form.save()
            return HttpResponseRedirect(reverse('projects'))
        
        project = form.save(commit=False)
        project.creator = request.user
        project.save()
        form.save_m2m()
        new_setting = Setting(user=request.user, project=project)
        new_setting.save()
        return HttpResponseRedirect(reverse('upload', args=[project.id]))

    return render(request, template_name, {
        'form': form
    })

class DatasetView(LoginRequiredMixin, ListView):
    template_name = 'admin/dataset.html'
    paginate_by = 10

    def get_queryset(self):
        project = get_object_or_404(Project, pk=self.kwargs['project_id'])
        return project.documents.all()

class AnnotationHistoryView(LoginRequiredMixin, TemplateView):
    template_name = 'admin/annotation_history.html'

class LabelView(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/label.html'

class StatsView(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/stats.html'


class SettingView(LoginRequiredMixin, TemplateView):
    template_name = 'admin/setting.html'
# TODO: Add ability to upload BiLSTMCRF document for RE
class DataUpload(SuperUserMixin, LoginRequiredMixin, TemplateView):
    template_name = 'admin/dataset_upload.html'
    class ImportFileError(Exception):
        def __init__(self, message):
            self.message = message

    def extract_metadata_csv(self, row, text_col, header_without_text):
        vals_without_text = [val for i, val in enumerate(row) if i != text_col]
        return json.dumps(dict(zip(header_without_text, vals_without_text)))

    def csv_to_documents(self, project, data_file, text_key='text', max_batch=500):
        decoded_data_file = TextIOWrapper(data_file, encoding='utf-8')
        reader = csv.reader(decoded_data_file)
        maybe_header = next(reader)
        if maybe_header:
            header = False
            docs = []
            if text_key in maybe_header:
                text_col = maybe_header.index(text_key)
                header_without_text = [title for i, title in enumerate(maybe_header) if i != text_col]
                header = True
            elif len(maybe_header) == 1:
                text_col = 0
                text = maybe_header[text_col]
                docs.append(Document(text=text, project=project))
            else:
                raise DataUpload.ImportFileError("CSV file must either have a column named '{}' or only have one column.".format(text_key))
            
            with transaction.atomic():
                for row in reader:
                    text = row[text_col]
                    metadata = self.extract_metadata_csv(row, text_col, header_without_text) if header else '\{\}'
                    docs.append(Document(text=text, metadata=metadata, project=project))
                    if len(docs) == max_batch:
                        Document.objects.bulk_create(docs)
                        docs = []
                Document.objects.bulk_create(docs)
        else:
            raise DataUpload.ImportFileError("CSV file is empty")

    def extract_metadata_json(self, entry):
        if "metadata" in entry:
            return json.dumps(entry["metadata"])
        else:
            temp = {}
            ignore_keys = set(["text", "annotations", "doc_id", "user"])
            for key in entry:
                if key not in ignore_keys:
                    temp[key] = entry[key]
            return json.dumps(temp)
    
    def create_plain_docs_from_json(self, project, data_file, text_key="text", max_batch=500):
        docs = []
        try:
            if data_file.multiple_chunks():
                data = ijson.items(data_file, "data.item")
            else:
                data = json.load(data_file)
                data = data["data"]
        except:
            raise DataUpload.ImportFileError("Document dictionaries do not have the 'data' key")
        
        with transaction.atomic():
            for entry in data:
                try:
                    text = entry[text_key]
                except:
                    raise DataUpload.ImportFileError("Document dictionaries do not have the '{}' key".format(text_key))
                
                metadata = self.extract_metadata_json(entry)
                
                docs.append(Document(text=text, metadata=metadata, project=project))
                if len(docs) == max_batch:
                    Document.objects.bulk_create(docs)
                    docs = []
            
            Document.objects.bulk_create(docs)

    def create_ner_docs_from_json(self, project, data_file, user, text_key="text"):
        all_labels = {}
        try:
            if data_file.multiple_chunks():
                data = ijson.items(data_file, "data.item")
            else:
                data = json.load(data_file)
                data = data["data"]
        except:
            raise DataUpload.ImportFileError("Document dictionaries do not have the 'data' key")
        
        with transaction.atomic():
            for entry in data:
                try:
                    text = entry[text_key]
                except:
                    raise DataUpload.ImportFileError("Document dictionaries do not have the '{}' key".format(text_key))

                metadata = self.extract_metadata_json(entry)
                
                current_doc = Document(text=text, metadata=metadata, project=project)
                current_doc.save()

                for annotation in entry["annotations"]: 
                    if annotation['label'] not in all_labels:
                        new_label = Label(text=annotation["label"], project=project, user_provided=True)
                        new_label.save()
                        all_labels[annotation["label"]] = new_label
                        
                    cur_annotation = Annotation(user_provided=True, task=project.task, user=user, document=current_doc, label=all_labels[annotation["label"]])
                    cur_annotation.save()
                    ner_annotation = NamedEntityAnnotation(annotation=cur_annotation, start_offset=annotation["start_offset"], end_offset=annotation["end_offset"])
                    ner_annotation.save()
            
    def post(self, request, *args, **kwargs):
        project = get_object_or_404(Project, pk=kwargs.get('project_id'))
        import_format = request.POST['format']
        upload_type = request.POST['upload_type']
        user = self.request.user
        try:
            data_file = request.FILES['dataset']
            if import_format == 'csv':
                self.csv_to_documents(project, data_file)
            elif import_format == 'json':
                if upload_type == "ner":
                    self.create_ner_docs_from_json(project, data_file, user)
                else:
                    self.create_plain_docs_from_json(project, data_file)
            return HttpResponseRedirect(reverse('dataset', args=[project.id]))
        except DataUpload.ImportFileError as e:
            messages.add_message(request, messages.ERROR, e.message)
            #We shouldn't refresh the page if there is an error
            #return HttpResponseRedirect(reverse('upload', args=[project.id]))
        except Exception as e:
            logger.exception(e)
            messages.add_message(request, messages.ERROR, 'Something went wrong')
            return HttpResponseRedirect(reverse('upload', args=[project.id]))


class DataDownload(LoginRequiredMixin, TemplateView):
    template_name = 'admin/dataset_download.html'


class DataDownloadFile(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        user_id = self.request.user.id
        project_id = self.kwargs['project_id']
        project = get_object_or_404(Project, pk=project_id)
        explanation_int = project.explanation_type
        export_format = request.GET.get('format')
        task_name = project.get_task_name()
        user_annotations = Annotation.objects.get_annotations_for_export(user_id, task_name, explanation_int)
        dataset = Document.objects.export_project_user_documents(task_name, project_id, user_id, export_format, user_annotations, explanation_int)
        filename = '_'.join(project.name.lower().split())
        try:
            if export_format == 'csv':
                response = self.get_csv(filename, dataset)
            elif export_format == 'json':
                response = self.get_json(filename, dataset)
            return response
        except Exception as e:
            logger.exception(e)
            messages.add_message(request, messages.ERROR, "Something went wrong")
            return HttpResponseRedirect(reverse('download', args=[project.id]))

    def get_csv(self, filename, dataset):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="{}.csv"'.format(filename)
        writer = csv.writer(response)
        writer.writerows(dataset)
        return response

    def get_json(self, filename, dataset):
        response = HttpResponse(content_type='text/json')
        response['Content-Disposition'] = 'attachment; filename="{}.json"'.format(filename)
        response.write(json.dumps(dataset, ensure_ascii=False, indent=1))
        return response


class LoginView(BaseLoginView):
    template_name = 'login.html'
    redirect_authenticated_user = True

