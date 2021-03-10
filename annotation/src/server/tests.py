# DISCLAIMER, THESE ARE NOT UP TO DATE IN ANYWAY, JUST LEFT AS A REFERENCE ON HOW TO WRITE TESTS
from django.test import TestCase
from django.contrib.auth.models import User
from django.core.exceptions import FieldError
import server.models as m


class TaskTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.task = m.Task.objects.create(name="test-task")
        cls.project = m.Project.objects.create(name="test-project", 
                                      description="",
                                      guideline="",
                                      task=cls.task)
                                
    def test_get_all_projects(self):
        projects = self.task.get_projects()
        self.assertEqual(projects.count(), 1)
        self.assertEqual(projects[0].name, "test-project")

class ProjectTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.task = m.Task.objects.create(name="Sentiment Analysis")
        cls.project = m.Project.objects.create(name="test-project", 
                                      description="",
                                      guideline="",
                                      task=cls.task)
        cls.doc_1 = m.Document.objects.create(text="test-1", project=cls.project)
        cls.doc_2 = m.Document.objects.create(text="test-2", project=cls.project)
        cls.doc_3 = m.Document.objects.create(text="test-3", project=cls.project)
    
    def test_get_absolute_url(self):
        url = self.project.get_absolute_url()
        self.assertEqual(url, "/projects/1/docs/create")
    
    def test_get_documents(self):
        docs = self.project.get_documents()
        self.assertEqual(docs.count(), 3)
        self.assertEqual(docs[0].text, "test-1")
    
    def test_get_annotated_documents(self):
        annotated_docs = self.project.get_annotated_documents()
        self.assertEqual(annotated_docs.count(), 0)

        self.doc_1.annotated = True
        self.doc_1.save()

        annotated_docs = self.project.get_annotated_documents()
        self.assertEqual(annotated_docs.count(), 1)
    
    def test_get_progress(self):
        progress_dict = self.project.get_progress()
        self.assertEqual(progress_dict["total"], 3)
        self.assertEqual(progress_dict["remaining"], 3)

        self.doc_2.annotated = True
        self.doc_2.save()

        progress_dict = self.project.get_progress()
        self.assertEqual(progress_dict["remaining"], 2)
    
    def test_get_task_name(self):
        task_name = self.project.get_task_name()
        self.assertEqual(task_name, "sentiment_analysis")

class DocumentTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create(username='Testuser')
        cls.task_1 = m.Task.objects.create(name="Sentiment Analysis")
        cls.project_1 = m.Project.objects.create(name="test-project-1", 
                                      description="",
                                      guideline="",
                                      task=cls.task_1)
        cls.doc_1 = m.Document.objects.create(text="test-1", project=cls.project_1)
        cls.label_1 = m.Label.objects.create(text="label-1", shortcut="a", project=cls.project_1)
        cls.label_2 = m.Label.objects.create(text="label-2", shortcut="b", project=cls.project_1)
        
        cls.annotation_1 = m.Annotation.objects.create(task=cls.task_1, document=cls.doc_1, user=cls.user, label=cls.label_1)
        m.SentimentAnalysisAnnotation.objects.create(annotation=cls.annotation_1)
        cls.annotation_2 = m.Annotation.objects.create(task=cls.task_1, document=cls.doc_1, user=cls.user, label=cls.label_1)
        m.SentimentAnalysisAnnotation.objects.create(annotation=cls.annotation_2)
        cls.annotation_3 = m.Annotation.objects.create(task=cls.task_1, document=cls.doc_1, user=cls.user, label=cls.label_2)
        m.SentimentAnalysisAnnotation.objects.create(annotation=cls.annotation_3)
        cls.annotation_4 = m.Annotation.objects.create(task=cls.task_1, document=cls.doc_1, user=cls.user, label=cls.label_2)
        m.SentimentAnalysisAnnotation.objects.create(annotation=cls.annotation_4)

        cls.task_2 = m.Task.objects.create(name="Named Entity Recognition")
        cls.project_2 = m.Project.objects.create(name="test-project-2", 
                                      description="",
                                      guideline="",
                                      task=cls.task_2)
        cls.doc_2 = m.Document.objects.create(text="Hi my name is Rahul", project=cls.project_2)
        cls.label_3 = m.Label.objects.create(text="label-3", shortcut="a", project=cls.project_2)
        cls.label_4 = m.Label.objects.create(text="label-4", shortcut="b", project=cls.project_2)

        cls.annotation_5 = m.Annotation.objects.create(task=cls.task_2, document=cls.doc_2, user=cls.user, label=cls.label_3)
        m.NamedEntityAnnotation.objects.create(annotation=cls.annotation_5, start_offset=0, end_offset=4)
        cls.annotation_6 = m.Annotation.objects.create(task=cls.task_2, document=cls.doc_2, user=cls.user, label=cls.label_4)
        m.NamedEntityAnnotation.objects.create(annotation=cls.annotation_6, start_offset=14, end_offset=19)

        cls.task_3 = m.Task.objects.create(name="Relation Extraction")
        cls.project_3 = m.Project.objects.create(name="test-project-1", 
                                      description="",
                                      guideline="",
                                      task=cls.task_3)
        cls.doc_3 = m.Document.objects.create(text="Good software always has tests", project=cls.project_3)
        cls.label_5 = m.Label.objects.create(text="label-5", shortcut="a", project=cls.project_3)
        cls.label_6 = m.Label.objects.create(text="label-6", shortcut="b", project=cls.project_3)
        
        cls.annotation_7 = m.Annotation.objects.create(task=cls.task_3, document=cls.doc_3, user=cls.user, label=cls.label_5)
        m.RelationExtractionAnnotation.objects.create(annotation=cls.annotation_7, start_offset_1=0, end_offset_1=4, start_offset_2=6, end_offset_2=14)
        cls.annotation_8 = m.Annotation.objects.create(task=cls.task_3, document=cls.doc_3, user=cls.user, label=cls.label_6)
        m.RelationExtractionAnnotation.objects.create(annotation=cls.annotation_8, start_offset_1=6, end_offset_1=14, start_offset_2=25, end_offset_2=30)

    def test_get_base_annotations(self):
        annotations = self.doc_1.get_base_annotations()
        self.assertEqual(annotations.count(), 4)
        self.assertEqual(annotations[0].label, self.label_1)
        self.assertEqual(annotations[3].task.name, "Sentiment Analysis")

        annotations = self.doc_2.get_base_annotations()
        self.assertEqual(annotations.count(), 2)
        self.assertEqual(annotations[0].label, self.label_3)
        self.assertEqual(annotations[1].task.name, "Named Entity Recognition")

        annotations = self.doc_3.get_base_annotations()
        self.assertEqual(annotations.count(), 2)
        self.assertEqual(annotations[0].label, self.label_5)
        self.assertEqual(annotations[1].task.name, "Relation Extraction")
    
    def test_get_extended_annotations_sa(self):
        extended_annotations = self.doc_1.get_extended_annotations()
        self.assertEqual(len(extended_annotations), 4)
        for annotation in extended_annotations:
            self.assertIsNotNone(extended_annotations[annotation])
            self.assertTrue(hasattr(extended_annotations[annotation], "annotation"))
    
    def test_get_extended_annotations_ner(self):
        extended_annotations = self.doc_2.get_extended_annotations()
        self.assertEqual(len(extended_annotations), 2)
        for annotation in extended_annotations:
            self.assertIsNotNone(extended_annotations[annotation])
            self.assertTrue(hasattr(extended_annotations[annotation], "start_offset"))
            self.assertTrue(hasattr(extended_annotations[annotation], "end_offset"))
    
    def test_get_extended_annotations_re(self):
        extended_annotations = self.doc_3.get_extended_annotations()
        self.assertEqual(len(extended_annotations), 2)
        for annotation in extended_annotations:
            self.assertIsNotNone(extended_annotations[annotation])
            self.assertTrue(hasattr(extended_annotations[annotation], "start_offset_1"))
            self.assertTrue(hasattr(extended_annotations[annotation], "end_offset_1"))
            self.assertTrue(hasattr(extended_annotations[annotation], "start_offset_2"))
            self.assertTrue(hasattr(extended_annotations[annotation], "end_offset_2"))
    
    def test_delete_annotations(self):
        self.doc_3.delete_annotations()
        self.assertEqual(self.doc_3.annotations.count(), 0)
    
    # Not needed right now, but placeholder
    def test_get_formatted_explanations_trigger(self):
        pass

    def test_get_formatted_explanations_nl(self):
        pass

    def test_get_raw_explanations_trigger(self):
        pass
    
    def test_get_raw_explanations_nl(self):
        pass
    



class AnnotationTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create(username='Testuser')
        cls.task_1 = m.Task.objects.create(name="Sentiment Analysis")
        cls.project_1 = m.Project.objects.create(name="test-project-1", 
                                      description="",
                                      guideline="",
                                      task=cls.task_1)
        cls.doc_1 = m.Document.objects.create(text="There are many positive good and great words", project=cls.project_1)
        cls.label_1 = m.Label.objects.create(text="label-1", shortcut="a", project=cls.project_1)
        cls.label_2 = m.Label.objects.create(text="label-2", shortcut="b", project=cls.project_1)
        
        cls.annotation_1 = m.Annotation.objects.create(task=cls.task_1, document=cls.doc_1, user=cls.user, label=cls.label_1)
        m.SentimentAnalysisAnnotation.objects.create(annotation=cls.annotation_1)
        m.TriggerExplanation.objects.create(annotation=cls.annotation_1, trigger_id=1, start_offset=0, end_offset=23)
        m.TriggerExplanation.objects.create(annotation=cls.annotation_1, trigger_id=2, start_offset=0, end_offset=14)
        m.TriggerExplanation.objects.create(annotation=cls.annotation_1, trigger_id=2, start_offset=24, end_offset=28)
        m.TriggerExplanation.objects.create(annotation=cls.annotation_1, trigger_id=3, start_offset=0, end_offset=14)
        m.TriggerExplanation.objects.create(annotation=cls.annotation_1, trigger_id=3, start_offset=33, end_offset=38)

        cls.task_2 = m.Task.objects.create(name="Named Entity Recognition")
        cls.project_2 = m.Project.objects.create(name="test-project-2", 
                                      description="",
                                      guideline="",
                                      task=cls.task_2)
        cls.doc_2 = m.Document.objects.create(text="Hi my name is Rahul", project=cls.project_2)
        cls.label_3 = m.Label.objects.create(text="label-3", shortcut="a", project=cls.project_2)
        cls.label_4 = m.Label.objects.create(text="label-4", shortcut="b", project=cls.project_2)

        cls.annotation_5 = m.Annotation.objects.create(task=cls.task_2, document=cls.doc_2, user=cls.user, label=cls.label_3)
        m.NamedEntityAnnotation.objects.create(annotation=cls.annotation_5, start_offset=0, end_offset=5)
        m.NaturalLanguageExplanation.objects.create(annotation=cls.annotation_5, text="text-1")
        m.NaturalLanguageExplanation.objects.create(annotation=cls.annotation_5, text="text-2")
        cls.annotation_6 = m.Annotation.objects.create(task=cls.task_2, document=cls.doc_2, user=cls.user, label=cls.label_4)
        m.NamedEntityAnnotation.objects.create(annotation=cls.annotation_6, start_offset=14, end_offset=19)
        m.NaturalLanguageExplanation.objects.create(annotation=cls.annotation_6, text="The word 'name' is less than 2 words to the left of PER")

        cls.task_3 = m.Task.objects.create(name="Relation Extraction")
        cls.project_3 = m.Project.objects.create(name="test-project-1", 
                                      description="",
                                      guideline="",
                                      task=cls.task_3)
        cls.doc_3 = m.Document.objects.create(text="Good software always has tests", project=cls.project_3)
        cls.label_5 = m.Label.objects.create(text="label-5", shortcut="a", project=cls.project_3)
        cls.label_6 = m.Label.objects.create(text="label-6", shortcut="b", project=cls.project_3)
        
        cls.annotation_7 = m.Annotation.objects.create(task=cls.task_3, document=cls.doc_3, user=cls.user, label=cls.label_5)
        m.RelationExtractionAnnotation.objects.create(annotation=cls.annotation_7, start_offset_1=0, end_offset_1=4, start_offset_2=6, end_offset_2=14)
        m.TriggerExplanation.objects.create(annotation=cls.annotation_7, trigger_id=1, start_offset=4, end_offset=5)
        cls.annotation_8 = m.Annotation.objects.create(task=cls.task_3, document=cls.doc_3, user=cls.user, label=cls.label_6)
        m.RelationExtractionAnnotation.objects.create(annotation=cls.annotation_8, start_offset_1=6, end_offset_1=14, start_offset_2=25, end_offset_2=30)
        m.TriggerExplanation.objects.create(annotation=cls.annotation_8, trigger_id=1, start_offset=21, end_offset=24)
        cls.annotation_9 = m.Annotation.objects.create(task=cls.task_3, document=cls.doc_3, user=cls.user, label=cls.label_6)
        m.RelationExtractionAnnotation.objects.create(annotation=cls.annotation_9, start_offset_1=6, end_offset_1=12, start_offset_2=15, end_offset_2=20)
        
    def test_get_extended_annotation_sa(self):
        extended_annotation = self.annotation_1.get_extended_annotation()
        self.assertEqual(extended_annotation.annotation.id, self.annotation_1.id)
    
    def test_get_extended_annotation_ner(self):
        extended_annotation = self.annotation_5.get_extended_annotation()
        self.assertEqual(extended_annotation.start_offset, 0)
        self.assertEqual(extended_annotation.end_offset, 5)
        self.assertEqual(extended_annotation.user_provided, False)
    
    def test_get_extended_annotation_re(self):
        extended_annotation = self.annotation_7.get_extended_annotation()
        self.assertEqual(extended_annotation.start_offset_1, 0)
        self.assertEqual(extended_annotation.end_offset_1, 4)
        self.assertEqual(extended_annotation.start_offset_2, 6)
        self.assertEqual(extended_annotation.end_offset_2, 14)
    
    def test_get_trigger_explanations(self):
        trigger_count = self.annotation_1._get_trigger_explanations().count()
        self.assertEqual(trigger_count, 5)

        with self.assertRaises(FieldError):
            self.annotation_5._get_trigger_explanations()

        with self.assertRaises(FieldError):
            self.annotation_6._get_trigger_explanations()

        trigger_count = self.annotation_7._get_trigger_explanations().count()
        self.assertEqual(trigger_count, 1)

        trigger_count = self.annotation_8._get_trigger_explanations().count()
        self.assertEqual(trigger_count, 1)
    
    def test_format_trigger_explanations(self):
        triggers = self.annotation_1._get_formatted_trigger_explanations()
        self.assertEqual(len(triggers), 3)
        self.assertEqual(triggers[0], "There are many positive")
        self.assertEqual(triggers[1], "There are many good")
        self.assertEqual(triggers[2], "There are many great")
        
        with self.assertRaises(FieldError):
            self.annotation_5._get_formatted_trigger_explanations()
        
        triggers = self.annotation_8._get_formatted_trigger_explanations()
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0], "has")
    
    def test_get_natural_language_explanations(self):
        with self.assertRaises(FieldError):
            self.annotation_1._get_natural_language_explantions()

        nl_count = self.annotation_5._get_natural_language_explantions().count()
        self.assertEqual(nl_count, 2)

        nl_count = self.annotation_6._get_natural_language_explantions().count()
        self.assertEqual(nl_count, 1)

        with self.assertRaises(FieldError):
            self.annotation_7._get_natural_language_explantions()

        with self.assertRaises(FieldError):
            self.annotation_8._get_natural_language_explantions()
    
    def test_format_natural_language_explanations(self):
        with self.assertRaises(FieldError):
            self.annotation_1._get_format_natural_language_explanations()

        explanations = self.annotation_5._get_format_natural_language_explanations()
        self.assertEqual(len(explanations), 2)
        self.assertEqual(explanations[0], "text-1")

        explanations = self.annotation_6._get_format_natural_language_explanations()
        self.assertEqual(len(explanations), 1)
        self.assertEqual(explanations[0], "The word 'name' is less than 2 words to the left of PER")
    
    def test_get_explanations_raw(self):
        self.assertEqual(self.annotation_1.get_explanations().count(), 5)
        self.assertEqual(self.annotation_5.get_explanations().count(), 2)
        self.assertIsNone(self.annotation_9.get_explanations())
    
    def test_get_explanations_formatted(self):
        self.assertEqual(len(self.annotation_1.get_explanations(True)), 3)
        self.assertEqual(len(self.annotation_5.get_explanations(True)), 2)
        self.assertIsNone(self.annotation_9.get_explanations(True))