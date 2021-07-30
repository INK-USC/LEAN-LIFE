- The frontend stores data in DB using django.
- Django also acts as a gateway for the frontend to asynchronous send training requests to FastAPI.
- The model training updates are sent back to django by FastAPI

# Table of contents

1. [Setup](#setup)
2. [URLs.py](#urls)
3. [Managers](#managers)
4. [Serializers](#serializers)

# Setup

1.  Please install [Postgres 12.3](http://postgresguide.com/setup/install.html) (in the linked example they use PostgreSQL 9.2, please ensure you replace 9.2 with 12.3)
    - If on linux, please make sure to start postgres `sudo service postgresql start`
    - if you use the installation guide above for unix or windows you shouldn't have to do this
2.  Ensure your `leanlife` environment is activated
3.  Open a terminal in the root directory execute `pip install -r requirements.txt`
4.  Execute `python -m spacy download en_core_web_sm` (or whatever version you'd like, just make sure to update this in [utils.py](server/utils.py#L8))
5.  We will now setup the postgres connection. So, ensure that postgres is up and running. Navigate to the `src` folder inside `annotation_backend`, execute `cd LEAN-LIFE/annotation_backend/src`.
    - Inside the `app` folder, navigate to [settings.py](app/settings.py#L100)
      - Find the `DATABASES` dictionary, and replace the `PASSWORD` value with your own password
6.  Navigate to the `src` folder inside `annotation_backend`, `cd LEAN-LIFE/annotation_backend/src` and run:
    - `./setup.sh PASSWORD-YOU-JUST-SET` <- (passing your password in as an argument)
    - you will be asked to create a user here, this user is what you will use to login to the LEAN-LIFE application
    - To create additional superuser/admin use django's `python manage.py createsuperuser`
7.  Start the backend from `cd LEAN-LIFE/annotation_backend/src` folder: `python manage.py runserver 0.0.0.0:8000`

### Potential Errors:

- Postgres is not installed:
  - To check: Open up terminal and exectue `which psql`. This should return a path.
  - To solve: Please follow the example [provided](http://postgresguide.com/setup/install.html)
- Some other application is listening on port 5432
  - To check: (Unix, Linux): `sudo lsof -i:5432`, (Windows): `netstat -tulpn | grep 5432`
    - [Useful Link](https://www.cyberciti.biz/faq/unix-linux-check-if-port-is-in-use-command/)
  - To solve: Get the Process ID of the application running on the port and kill the process.
    - [Windows](https://www.revisitclass.com/networking/how-to-kill-a-process-which-is-using-port-8080-in-windows/)
    - [Unix, Linux––2nd Answer](https://stackoverflow.com/questions/3855127/find-and-kill-process-locking-port-3000-on-mac)

# URLs

- Paths starting with `api/` are used by Vue.js frontend to save and retrieve data.
- Other paths are for supporting django templates, which might be removed in a future release.

# Managers

Once we have Data Models, Django gives us a database abstraction API that lets us create, retrieve, update, and delete objects.
A model class represents database table, and an instance of that class represents a particular record in the database table.
We use a similar structure for our models. For more information, visit [this](https://docs.djangoproject.com/en/3.2/topics/db/managers/) page.

# Serializers

We use serailizer to convert the querysets and model instances to native Python datatypes that can be easily rendered into JSON, XML, etc.
We use different serializer like `NaturalLanguageExplanationSerializer`, `SingleUserAnnotatedDocumentExplanationsSerializer`, `SentimentAnalysisAnnotationSerializer`, `RelationExtractionAnnotationHistorySerializer` etc.
