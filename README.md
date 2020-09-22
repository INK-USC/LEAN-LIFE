# LEAN-LIFE
**L**abel **E**fficient **A**nnotatio**N** framework that allows for **L**earn**I**ng **F**rom **E**xplanations

[Website](http://inklab.usc.edu/leanlife/)  &nbsp;&nbsp;&nbsp;&nbsp;
[Paper](https://arxiv.org/abs/2004.07499) &nbsp;&nbsp;&nbsp;&nbsp;

### Contents:
* [Quick Intro](#quick-intro)
* [Release Plan](#release-plan)
* [Installation Instructions](#installation-instructions)
* [Set Up Instructions](#set-up-instructions)
* [How To Use](#how-to-use)
* [Supported Data Formats](#supported-data-formats)
* [Contributing](#contributing)
* [Citation](#citation)

# Quick Intro:
LEAN-LIFE is an annotation framework for Named Entity Recognition (NER), Relation Extraction (RE) and Sentiment Analysis (SA)/Multi-Class Document Classification. LEAN-LIFE additionally enable the capture and use of explanations during the annotation process. Explanations can be seen as enhanced supervision, providing reasoning behind the labeling decision.

Our initial frontend code is based on the [Doccano](https://github.com/chakki-works/doccano) project and the [AlpacaTag](https://github.com/INK-USC/AlpacaTag) project however we differentiate ourselves in these ways:

* **Triplet Capture**: Allows the building of a dataset that is (datapoint, label, explanation), unlike the standard (datapoint, label) tuple.

* **Explanation Supported Recommendations**: A backend soft-matching model is updated in a batch fashion to provide recommendations to the user. (coming soon)

* **Relation Extraction Task Supported**: Using the output of the Named Entity Extraction task, our system allows for the creation of relation extraction datasets.

Due to refactoring efforts and a desire to create a more stable framework the following features are not supported yet, but will be supported soon:

* **Active intelligent recommendation**: Instead of just using explanations, we will be training an appropriate backend model for the selected task using explanations and labels to both provide enhanced annotations to the user and ensure users are not asked to provide annotations on documents that the model already understands.

* **Real-time model deployment**: Users can extract the trained recommendation model at any point for deployment purposes, without having to wait till all documents are labeled.

* **User Roles**: Differentiating between a project creator and project annotators, allowing for a creator to set up a project, while allowing annotators to configure more local settings like what types of recommendations they would like, and how often their backend model should be trained.

Please reference our [website](http://inklab.usc.edu/leanlife/) for more information.

We strongly encourage community engagement, please refer to our [contribution section](#contributing) for more on how to contribute!

# Release Plan:
**Next Release's Goals:**

* Goal 1
* Goal 2
* Goal 3

**Release 1** (Date: fill-this-in)

This release focuses on providing a Web-UI to annotate sets of documents. As mentioned the user can create annotations for three tasks (NER, SA, RE), as well as provide explanations for those annotations in two different ways. We support various import/export formats, and take advantage of two recommendation strategies for NER (Noun phrase chunking--powered by spaCy, Historical Annotation Application) and one for RE (Historical Annotation Application). A project owner can upload a set of (x,y) pairs as "Historical" annotations, and we will apply these pairs as recommendations for un-annotated documents.
 
# Installation Instructions:

Note: All paths are relative to being just outside the `LEAN-LIFE` directory. Please adjust paths accordingly.

* Please install [Python 3.6.5](https://www.python.org/downloads/release/python-365/) (if you use `conda` you can ignore this step)
* Please intall [Postgres 12.3](http://postgresguide.com/setup/install.html) (in the linked example they use PostgreSQL 9.2, please ensure you replace 9.2 with 12.3)
* Clone this repo: `git clone git@github.com:INK-USC/LEAN-LIFE.git`
* Create a virtual environment using:
     * annaconda: `conda create -n leanlife python==3.6` (annaconda doesn't have a stable 3.6.5 version)
     * virtualenv:
          1. `python3.6.5 -m pip install virtualenv`
          2. `python3.6.5 -m venv leanlife`
* Activate your environment:
     * annaconda: `conda activate leanlife`
     * virtualenv: `source leanlife/bin/activate`
* `pip install -r requirements.txt` 
* `python -m spacy download en` (or whatever version you'd like, just make sure to update this in [utils.py](https://github.com/INK-USC/LEAN-LIFE/blob/master/annotation/src/server/utils.py#L8))

-----

#### Potential Errors:
  * Postgres is not installed:
    - To check: Open up terminal and exectue `which psql`. This should return a path.
    - To solve: Please follow the example [provided](http://postgresguide.com/setup/install.html)
  * Some other application is listening on port 5432
    * To check: (Unix, Linux): `sudo lsof -i:5432`, (Windows): `netstat -tulpn | grep 5432`
      - [Useful Link](https://www.cyberciti.biz/faq/unix-linux-check-if-port-is-in-use-command/)
    * To solve: Get the Process ID of the application running on the port and kill the process.
      * [Windows](https://www.revisitclass.com/networking/how-to-kill-a-process-which-is-using-port-8080-in-windows/)
      * [Unix, Linux––2nd Answer](https://stackoverflow.com/questions/3855127/find-and-kill-process-locking-port-3000-on-mac)
  * Wrong version of python is being used.
    * To check: if you're getting installation errors, it could be that your machine is running the wrong version of python and/or installed packages. To check run `which python` and make sure the returned folder is the path to the `leanlife` virtual environment folder. To check that python is looking in the right places check this example [here](https://bic-berkeley.github.io/psych-214-fall-2016/sys_path.html#python-looks-for-modules-in-sys-path). Again the path should be the site-packages folder in your `leanlife` virtual environment
    * To Fix: Re-create virtual environment:
      - `deactivate leanlife`
      - `rm -rf leanlife`
      - make sure no other virtualenvs are running
        + open up terminal/command prompt and see if there are paranthesis at the start of each line, ex: `(base) user@...`
        + if this is the case deactivate that environment: `deactivate environment-name`, in the above example it would be `deactivate base`
      - Go to step 4 of installation instructions


# Set Up Instructions:

Note: All paths are relative to being just outside the `LEAN-LIFE` directory. Please adjust paths accordingly. We will dockerize this project soon as well.

1. Ensure your `leanlife` environment is activated
2. Navigate to the `server` folder inside `annotation/src`, `cd LEAN-LIFE/annotation/src/server`
3. `npm install`
4. `npm run build`
5. Navigate to the `src` folder inside `annotation`, `cd LEAN-LIFE/annotation/src`
     * Inside the `app` folder, navigate to [settings.py](fill-this-in) (We are setting up the postgres connection)
          * Find the `DATABASES` dictionary, and set a `PASSWORD` to your liking
6. Navigate to the `src` folder inside `annotation`, `cd LEAN-LIFE/annotation/src` and run `./setup.sh PASSWORD-YOU-JUST-SET`
     * you will be asked to create a user here, this user is what you will use to login to the LEAN-LIFE application
7. `python manage.py runserver 0.0.0.0:8000`
8. Open up an browser window and navigate to http://0.0.0.0:8000/

# How To Use:

### How to create a Project:

1. Log in using the user you created in step 6 of the set up.
2. Hit the *Create Project* button
3. Fill Project Fields
     * Choose the Project type (NER, RE, SA)
     * Explanation Type (Natural Language or Trigger)
     * Assigned Users
     * Name, Description
4. Upload documents per the presented format (csv and JSON formats supported, JSON is preferred for text parsing reasons, splitting on `,` isn't great)
5. Create the desired labels
6. Set Annotation Settings
     * Recommendation Type (Historical (NER, RE), Noun Phrase (NER))
     * Acquire Size (How many documents should be gotten from the db at a time)
7. (Optional) Upload annotations to be used as recommendations to annotators when appropraite, `Historical` recommendations
8. Start Annotating

### How to create addtional users:
Superuser

1. `cd annotation/src`
2. `python manage.py createsuperuser`

Normal User

1. `cd annotation/src`
2. `python manage.py shell`
     
     (a shell will pop up now, enter these commands)

     ```
     from django.contrib.auth.models import User
     user=User.objects.create_user('fill-username-in',  password='fill-password-in')
     user.save()
     ```

### How to annotate for the NER Task:

* Option 1: Highlight a word/phrase (span) in the presented text and select one of the Project's labels appearing just above the text.

* Option 2: Click on the provided recommendations in the `Recommendation Section`. These recommendations can be for both span detection and the appropriate label for the span (Historical, Explanation Soft Matching), or just span detection (Noun Phrase).

* If you have an explanation type set, the appropriate pop-up will appear to capture the explanation.

### How to annotate for the RE Task:

* Select two entities, the first entity is the **Subject** of the Relation (the entity that is more central to the relation), while the second entity is the **Object** of the Relation (the entity that is associated with the subject). 
     * Example: **John** (Subject) is born in **May** (Object).

* At this point if recommendations are turned on, recommended labels are highlighted in red. Else just select the appropriate relation label.

* If you have an explanation type set, the appropriate pop-up will appear to capture the explanation.

### How to annotate for the SA Task:

* If recommendations are turned on, recommended labels are highlighted in red, otherwise simply select a label for the provided text.

* If you have an explanation type set, the appropriate pop-up will appear to capture the explanation.

### How to provide a Trigger Explanation:

* Triggers can be seen as an extractive form of explanations, where you select spans from the text that informs a person when they make a labeling decision.

* Per annotation we allow for the capture of up to 4 triggers
     * Triggers cannot overlap though

* Per trigger, spans do not have to be consecutive

* If you have selected the Trigger Explanation Type for your Project, the Trigger popup will appear after you make an annotation.
     * If you do not choose to leave an explanation for this annotation, you may hit the `x` button at the top right
     * If you do wish to leave an explanation, select the spans of text that helped you make your labeling decision
     * Once you have completed the first trigger, you can hit the plus button to create a new trigger
     * Distinct phrases/clues should appear as different triggers
          * Ex: "had dinner at", "the food", "spicy" should be three separate triggers for the example text: "I had a fantastic dinner at Xi'an Famous Foods, the food is always fresh and spicy!". These triggers would be associated with the annotation that "Xi'an Famous Foods" is a `restaurant`. 

* Another Example:
     * Text: Louis Armstrong, the great trumpet player, lived in Corona. 
     * Annotation: Corona is a LOC(ation)
     * Trigger: "lived in"

### How to provide a Natural Language Explanation:

* Natural Language Explanations are written out reasons as to why you made a certain labeling decision. These reasons though must be parsable by the [NExT](https://github.com/INK-USC/NExT) parser, but we provide appropriate examples per task to help users write usable explanations.

* Per annotation we allow for the capture of up to 4 NL explanations

* If you have selected the Natural Language Explanation Type for your Project, the NL popup will appear after you make an annotation.
     * If you do not choose to leave an explanation for this annotation, you may hit the `x` button at the top right
     * If you do wish to leave an explanation, look through the available templates and fill in the blanks as you see fit
     * Once you have completed the first explanation, you can hit the plus button to create a new explanation
     * Distinct ideas/clues should appear as different nl explanations
          * If you want to use AND in your explanation, we request that you create another explanation instead

* Example:
     * Text: Louis Armstrong, the great trumpet player, lived in Corona.
     * Annotation: **Louis Armstrong**'s occupation is a **trumpet player**
     * NL Explanation: 
          1. The token  ','  appears between 'Louis Armstrong' and 'trumpet player'
          2. The token ',' appears to the right of 'trumpet player' by no more than 2 words
          3. There are no more than **5** words between 'Louis Armstrong' and 'trumpet player'
     
# Supported Data Formats

### NER:

**Import**

* JSON (recommended):
     ```
     {
          "data" : [
               {
                    "text" : "abcd",
                    "foo" : "bar"
               },
               {
                    "text" : "efgh",
                    "foo" : "man"
               },
               ...
          ]
     }
     ```
     * Each entry within `data` must have a key `text`. All other keys will be saved in a metadata dictionary associated with the text
* CSV:
     * Two formats are acceptable (but file must be using utf-8 encoding):
          1. With a header row, a column name must be text. All other columns will be saved in a metadata dictionary associated with the text
          2. No header, single column file with just text

     *  No commas can be in your text, which is why we strongly recommend using our json import process

**Export:**
* JSON:
     ```
     {
          "data": [
               {
                    "doc_id": 25,
                    "text": "Louis Armstrong, the great trumpet player, lived in Corona.",
                    "annotations": [
                         {
                              "annotation_id": 18,
                              "label": "LOC",
                              "start_offset": 52,
                              "end_offset": 58,
                              "explanation": [
                                   "lived in"
                              ]
                         }
                         ...
                    ],
                    "user": 4,
                    "metadata": {"foo" : "bar"}
               }
               ...
          ]
    }
     ```
* CSV:

     * Extended BIO format
     
     | document_id | word      | label | metadata | explanation                 |
     |-------------|-----------|-------|----------|-----------------------------|
     | 1           | Louis     | B-PER | {}       | , the great trumpet player, |
     | 1           | Armstrong | I-PER |          |                             |
     | 1           | ,         | O     |          |                             |

### RE:

**Import:**
* In order to create an RE Project, you must have already have annotated the Named Entities in your documents.

* JSON:
     - Essentially the output of an NER Project
     ```
     {
       "data": [
           {
                 "text": "Louis Armstrong, the great trumpet player, lived in Corona.",
                 "annotations": [
                     {
                           "label": "LOC",
                           "start_offset": 52,
                           "end_offset": 58
                     }
                     ...
                 ],
                 "metadata": {"foo" : "bar"}
           }
           ...
       ]
     }
     ```

**Export:**
 * JSON:
     ```
     {
          "data": [
               {
                    "doc_id": 25,
                    "text": "Louis Armstrong, the great trumpet player, lived in Corona.",
                    "annotations": [
                         {
                              "annotation_id": 18,
                              "label": "per:occupation",
                              "sbj_start_offset": 0,
                              "sbj_end_offset": 15,
                              "obj_start_offset": 27,
                              "obj_end_offset": 41,
                              "explanation": [
                                   "The token  ','  appears between 'Louis Armstrong' and 'trumpet player'", 
                                   "The token ',' appears to the right of 'trumpet player' by no more than 2 words",
                                   "There are no more than 5 words between 'Louis Armstrong' and 'trumpet player'"
                              ]
                         }
                         ...
                    ],
                    "user": 4,
                    "metadata": {"foo" : "bar"}
               }
               ...
          ]
    }
     ```
* CSV:
     
| document_id | entity_1        | entity_2       | label          | metadata | explanation                                                                                     |
|-------------|-----------------|----------------|----------------|----------|-------------------------------------------------------------------------------------------------|
| 1           | Louis Armstrong | trumpet player | per:occupation | {}       | The token  ','  appears between 'Louis Armstrong' and 'trumpet player':\*:\*:The token ',' ... |
|             |                 |                |                |          |                                                                                                 |
* Where ":\*:\*:" is a separator to split up the string in the explanation column. A workaround for the problem of splitting on ","

### SA
**Import**
* Same as NER formats

**Export**
* JSON:
     ```
     {
          "data": [
               {
                    "doc_id": 25,
                    "text": "Louis Armstrong, the great trumpet player, lived in Corona.",
                    "annotations": [
                         {
                              "annotation_id": 18,
                              "label": "postive",
                              "explanation": [
                                   "The word 'great' appears in the sentence."
                              ]
                         }
                         ...
                    ],
                    "user": 4,
                    "metadata": {"foo" : "bar"}
               }
               ...
          ]
    }
     ```

# Contributing

We love contributions, so thank you for taking the time! Pusing changes to master is blocked, so please create a branch and make your edits on the branch. Once done, please create a Pull Request and ask a contributer from the INK-LAB to pull your changes in. You can refer to our PR guidelines and general contribution guidelines [here](./CONTRIBUTING.md).

# Citation
```
@inproceedings{
    LEANLIFE2020,
    title={LEAN-LIFE: A Label-Efficient Annotation Framework Towards Learning from Explanation},
    author={Lee, Dong-Ho and Khanna, Rahul and Lin, Bill Yuchen and Chen, Jamin and Lee, Seyeon and Ye, Qinyuan and Boschee, Elizabeth and Neves, Leonardo and Ren, Xiang},
    booktitle={Proc. of ACL (Demo)},
    year={2020},
    url={}
}
```
