# Generated by Django 2.1.7 on 2020-03-27 19:51

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Annotation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('prob', models.FloatField(default=0.0)),
                ('via_recommendation', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('annotated', models.BooleanField(default=False)),
                ('metadata', models.TextField(default='{}')),
            ],
        ),
        migrations.CreateModel(
            name='Label',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.CharField(max_length=30)),
                ('shortcut', models.CharField(blank=True, choices=[('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd'), ('e', 'e'), ('f', 'f'), ('g', 'g'), ('h', 'h'), ('i', 'i'), ('j', 'j'), ('k', 'k'), ('l', 'l'), ('m', 'm'), ('n', 'n'), ('o', 'o'), ('p', 'p'), ('q', 'q'), ('r', 'r'), ('s', 's'), ('t', 't'), ('u', 'u'), ('v', 'v'), ('w', 'w'), ('x', 'x'), ('y', 'y'), ('z', 'z'), ('ctrl a', 'ctrl a'), ('ctrl b', 'ctrl b'), ('ctrl c', 'ctrl c'), ('ctrl d', 'ctrl d'), ('ctrl e', 'ctrl e'), ('ctrl f', 'ctrl f'), ('ctrl g', 'ctrl g'), ('ctrl h', 'ctrl h'), ('ctrl i', 'ctrl i'), ('ctrl j', 'ctrl j'), ('ctrl k', 'ctrl k'), ('ctrl l', 'ctrl l'), ('ctrl m', 'ctrl m'), ('ctrl n', 'ctrl n'), ('ctrl o', 'ctrl o'), ('ctrl p', 'ctrl p'), ('ctrl q', 'ctrl q'), ('ctrl r', 'ctrl r'), ('ctrl s', 'ctrl s'), ('ctrl t', 'ctrl t'), ('ctrl u', 'ctrl u'), ('ctrl v', 'ctrl v'), ('ctrl w', 'ctrl w'), ('ctrl x', 'ctrl x'), ('ctrl y', 'ctrl y'), ('ctrl z', 'ctrl z'), ('shift a', 'shift a'), ('shift b', 'shift b'), ('shift c', 'shift c'), ('shift d', 'shift d'), ('shift e', 'shift e'), ('shift f', 'shift f'), ('shift g', 'shift g'), ('shift h', 'shift h'), ('shift i', 'shift i'), ('shift j', 'shift j'), ('shift k', 'shift k'), ('shift l', 'shift l'), ('shift m', 'shift m'), ('shift n', 'shift n'), ('shift o', 'shift o'), ('shift p', 'shift p'), ('shift q', 'shift q'), ('shift r', 'shift r'), ('shift s', 'shift s'), ('shift t', 'shift t'), ('shift u', 'shift u'), ('shift v', 'shift v'), ('shift w', 'shift w'), ('shift x', 'shift x'), ('shift y', 'shift y'), ('shift z', 'shift z'), ('ctrl shift a', 'ctrl shift a'), ('ctrl shift b', 'ctrl shift b'), ('ctrl shift c', 'ctrl shift c'), ('ctrl shift d', 'ctrl shift d'), ('ctrl shift e', 'ctrl shift e'), ('ctrl shift f', 'ctrl shift f'), ('ctrl shift g', 'ctrl shift g'), ('ctrl shift h', 'ctrl shift h'), ('ctrl shift i', 'ctrl shift i'), ('ctrl shift j', 'ctrl shift j'), ('ctrl shift k', 'ctrl shift k'), ('ctrl shift l', 'ctrl shift l'), ('ctrl shift m', 'ctrl shift m'), ('ctrl shift n', 'ctrl shift n'), ('ctrl shift o', 'ctrl shift o'), ('ctrl shift p', 'ctrl shift p'), ('ctrl shift q', 'ctrl shift q'), ('ctrl shift r', 'ctrl shift r'), ('ctrl shift s', 'ctrl shift s'), ('ctrl shift t', 'ctrl shift t'), ('ctrl shift u', 'ctrl shift u'), ('ctrl shift v', 'ctrl shift v'), ('ctrl shift w', 'ctrl shift w'), ('ctrl shift x', 'ctrl shift x'), ('ctrl shift y', 'ctrl shift y'), ('ctrl shift z', 'ctrl shift z'), ('', '')], max_length=15, null=True)),
                ('background_color', models.CharField(default='#209cee', max_length=7)),
                ('text_color', models.CharField(default='#ffffff', max_length=7)),
            ],
        ),
        migrations.CreateModel(
            name='NamedEntityAnnotation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start_offset', models.PositiveIntegerField()),
                ('end_offset', models.PositiveIntegerField()),
                ('user_provided', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='NamedEntityAnnotationHistory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.TextField(max_length=100)),
                ('label', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='server.Label')),
            ],
        ),
        migrations.CreateModel(
            name='NaturalLanguageExplanation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField(max_length=500, validators=[django.core.validators.MinLengthValidator(1)])),
            ],
        ),
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('description', models.CharField(max_length=500)),
                ('guideline', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='RelationExtractionAnnotation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start_offset_1', models.PositiveIntegerField()),
                ('end_offset_1', models.PositiveIntegerField()),
                ('start_offset_2', models.PositiveIntegerField()),
                ('end_offset_2', models.PositiveIntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='RelationExtractionAnnotationHistory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word_1', models.TextField(max_length=100)),
                ('word_2', models.TextField(max_length=100)),
                ('label', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='server.Label')),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='relation_extraction_history', to='server.Project')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Setting',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedding', models.IntegerField()),
                ('nounchunk', models.BooleanField()),
                ('onlinelearning', models.BooleanField()),
                ('history', models.BooleanField()),
                ('active', models.IntegerField()),
                ('batch', models.IntegerField()),
                ('epoch', models.IntegerField()),
                ('acquire', models.IntegerField()),
                ('explanation', models.IntegerField()),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='settings', to='server.Project')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Task',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(choices=[('SentimentAnalysis', 'Sentiment Analysis'), ('NamedEntityRecognition', 'Named Entity Recognition'), ('RelationExtraction', 'Relation Extraction')], max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='TriggerExplanation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('trigger_id', models.PositiveIntegerField()),
                ('start_offset', models.PositiveIntegerField()),
                ('end_offset', models.PositiveIntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='SentimentAnalysisAnnotation',
            fields=[
                ('annotation', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, related_name='is_sa', serialize=False, to='server.Annotation')),
            ],
        ),
        migrations.AddField(
            model_name='triggerexplanation',
            name='annotation',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='trigger_explanations', to='server.Annotation'),
        ),
        migrations.AddField(
            model_name='relationextractionannotation',
            name='annotation',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='relation_extraction_annotation', to='server.Annotation'),
        ),
        migrations.AddField(
            model_name='project',
            name='task',
            field=models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, related_name='projects', to='server.Task'),
        ),
        migrations.AddField(
            model_name='project',
            name='users',
            field=models.ManyToManyField(related_name='projects', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='naturallanguageexplanation',
            name='annotation',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='natural_language_explanations', to='server.Annotation'),
        ),
        migrations.AddField(
            model_name='namedentityannotationhistory',
            name='project',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='named_entity_history', to='server.Project'),
        ),
        migrations.AddField(
            model_name='namedentityannotationhistory',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='namedentityannotation',
            name='annotation',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='named_entity_annotation', to='server.Annotation'),
        ),
        migrations.AddField(
            model_name='label',
            name='project',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='labels', to='server.Project'),
        ),
        migrations.AddField(
            model_name='document',
            name='project',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='documents', to='server.Project'),
        ),
        migrations.AddField(
            model_name='annotation',
            name='document',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='annotations', to='server.Document'),
        ),
        migrations.AddField(
            model_name='annotation',
            name='label',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='annotations', to='server.Label'),
        ),
        migrations.AddField(
            model_name='annotation',
            name='task',
            field=models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, related_name='annotations', to='server.Task'),
        ),
        migrations.AddField(
            model_name='annotation',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, related_name='annotations', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddIndex(
            model_name='triggerexplanation',
            index=models.Index(fields=['annotation', 'trigger_id'], name='trigger_annotation_index'),
        ),
        migrations.AlterUniqueTogether(
            name='triggerexplanation',
            unique_together={('annotation', 'start_offset', 'end_offset')},
        ),
        migrations.AlterUniqueTogether(
            name='setting',
            unique_together={('user', 'project')},
        ),
        migrations.AddIndex(
            model_name='relationextractionannotationhistory',
            index=models.Index(fields=['project', 'user'], name='re_history_lookup_index'),
        ),
        migrations.AlterUniqueTogether(
            name='relationextractionannotationhistory',
            unique_together={('user', 'word_1', 'word_2', 'label')},
        ),
        migrations.AddIndex(
            model_name='relationextractionannotation',
            index=models.Index(fields=['annotation'], name='re_annotation_index'),
        ),
        migrations.AddIndex(
            model_name='naturallanguageexplanation',
            index=models.Index(fields=['annotation'], name='nl_annotation_index'),
        ),
        migrations.AlterUniqueTogether(
            name='naturallanguageexplanation',
            unique_together={('annotation', 'text')},
        ),
        migrations.AddIndex(
            model_name='namedentityannotationhistory',
            index=models.Index(fields=['project', 'user'], name='ne_history_lookup_index'),
        ),
        migrations.AlterUniqueTogether(
            name='namedentityannotationhistory',
            unique_together={('user', 'word', 'label')},
        ),
        migrations.AddIndex(
            model_name='namedentityannotation',
            index=models.Index(fields=['annotation'], name='named_entity_annotation_index'),
        ),
        migrations.AlterUniqueTogether(
            name='label',
            unique_together={('project', 'shortcut'), ('project', 'text')},
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['project'], name='document_project_index'),
        ),
        migrations.AddIndex(
            model_name='annotation',
            index=models.Index(fields=['user'], name='annotation_user_index'),
        ),
        migrations.AddIndex(
            model_name='annotation',
            index=models.Index(fields=['document', 'user', 'label'], name='annotation_retrieval_index'),
        ),
        migrations.RunSQL("INSERT INTO server_task (name) VALUES ('Sentiment Analysis')"),
        migrations.RunSQL("INSERT INTO server_task (name) VALUES ('Named Entity Recognition')"),
        migrations.RunSQL("INSERT INTO server_task (name) VALUES ('Relation Extraction')")
    ]
