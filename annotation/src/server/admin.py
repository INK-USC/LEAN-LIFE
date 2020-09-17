from django.contrib import admin

from .models import Label, Document, Project
from .models import NamedEntityAnnotation

class LabelAdmin(admin.ModelAdmin):
    list_display = ('text', 'project', 'text_color', 'background_color')
    ordering = ('project',)
    search_fields = ('project',)


class DocumentAdmin(admin.ModelAdmin):
    list_display = ('text', 'project', 'annotated', 'metadata')
    ordering = ('project',)
    search_fields = ('project',)


class ProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'task')
    ordering = ('task',)
    search_fields = ('name',)


# class NamedEntityAnnotation(admin.ModelAdmin):
#     list_display = ('annotation', 'start_offset', 'end_offset', 'user_provided')
#     ordering = ('annotation',)
#     search_fields = ('annotation',)

# class DocumentAnnotationAdmin(admin.ModelAdmin):
#     list_display = ('document', 'label', 'user')
#     ordering = ('document',)
#     search_fields = ('document',)
#
#
# class Seq2seqAnnotationAdmin(admin.ModelAdmin):
#     list_display = ('document', 'text', 'user')
#     ordering = ('document',)
#     search_fields = ('document',)

#admin.site.register(DocumentAnnotation, DocumentAnnotationAdmin)
# admin.site.register(NamedEntityAnnotation, SequenceAnnotationAdmin)
#admin.site.register(Seq2seqAnnotation, Seq2seqAnnotationAdmin)
admin.site.register(Label, LabelAdmin)
admin.site.register(Document, DocumentAdmin)
admin.site.register(Project, ProjectAdmin)
