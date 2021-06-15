- The frontend stores data in DB using django.
- Django also acts as a gateway for the frontend to asynchronous send training requests to FastAPI.
- The model training updates are sent back to django by FastAPI

# Table of contents

1. [URLs.py](#urlspy)
2. [Managers](#managers)
3. [Serializers](#serializers)

# URLs.py

- Paths starting with `api/` are used by Vue.js frontend to save and retrieve data.
- Other paths are for supporting django templates, which might be removed in a future release.

# Managers

Once we have Data Models, Django gives us a database abstraction API that lets us create, retrieve, update, and delete objects.
A model class represents database table, and an instance of that class represents a particular record in the database table.
We use a similar structure for our models. For more information, visit [this](https://docs.djangoproject.com/en/3.2/topics/db/managers/) page.

# Serializers

We use serailizer to convert the querysets and model instances to native Python datatypes that can be easily rendered into JSON, XML, etc.
We use different serializer like `NaturalLanguageExplanationSerializer`, `SingleUserAnnotatedDocumentExplanationsSerializer`, `SentimentAnalysisAnnotationSerializer`, `RelationExtractionAnnotationHistorySerializer` etc.
