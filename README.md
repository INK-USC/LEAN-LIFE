# LEAN-LIFE
**L**abel **E**fficient **A**nnotatio**N** framework that allows for **L**earn**I**ng **F**rom **E**xplanations

(Staging Branch)

[Website](http://inklab.usc.edu/leanlife/)  &nbsp;&nbsp;&nbsp;&nbsp;
[Paper](https://arxiv.org/abs/2004.07499) &nbsp;&nbsp;&nbsp;&nbsp;

### Contents:
* [Quick Intro](#quick-intro)
* [Release Plan](#release-plan)
* [Installation Instructions](#installation-instructions)
* [Set Up Instructions](#set-up-instructions)
* [Contributing](#contributing)
* [Misc](#misc)
* [Citation](#citation)

# Quick Intro:
LEAN-LIFE is an annotation framework for Named Entity Recognition (NER), Relation Extraction (RE) and Sentiment Analysis (SA)/Multi-Class Document Classification. LEAN-LIFE additionally enables the capture and use of explanations during the annotation process. Explanations can be seen as enhanced supervision, providing reasoning behind the labeling decision, and thus help speed up model training in low-resource environments.

Our initial frontend code is based on the [Doccano](https://github.com/chakki-works/doccano) project and the [AlpacaTag](https://github.com/INK-USC/AlpacaTag) project however we differentiate ourselves in these ways:

* **Triplet Capture**: Allows the building of a dataset that is (datapoint, label, explanation), unlike the standard (datapoint, label) tuple.

* **Explanation Supported Model Training**: Taining of both [TriggerNER](https://github.com/INK-USC/TriggerNER) and [NExT](https://github.com/INK-USC/NExT) models, for both model deployment and recommendations (coming soon)

* **Relation Extraction Task Supported**: Using the output of the Named Entity Extraction task, our system allows for the creation of relation extraction datasets.

Due to refactoring efforts and a desire to create a more stable framework the following features are not supported yet, but will be supported soon:

* **Active intelligent recommendation**: Iteratively train an appropriate backend model for the selected task using explanations and labels to both provide enhanced annotations to annotators and ensure annotators are not asked to provide annotations on documents that the model already understands.

* **Real-time model extraction**: Users can extract the trained recommendation model at any point for deployment purposes, without having to wait till all documents are labeled.

* **Model Interaction API**: Seperate API for model training (batch), prediction(online and batch), and extraction--this functionality will be built seperately from our annotation framework

* **User Roles**: Differentiating between a project creator and project annotators, allowing for a creator to set up a project, while allowing annotators to configure more local settings like what types of recommendations they would like, and how often their backend model should be trained.

Reference our [website](http://inklab.usc.edu/leanlife/) for more information.

For information on how to use the annotation framework and supported data formats, please look at our [wiki](https://github.com/INK-USC/LEAN-LIFE/wiki) 

We strongly encourage community engagement, please refer to our [contribution section](#contributing) for more on how to contribute!

# Release Plan:

**Next Release's Goals:** 4-6 Weeks

* API for training of a downstream model (RE, SA):
  * with natural language explanations, a fixed NExT parser to generate weak labels (RE, SA)
* UI to allow for training and downloading of mentioned models after annotations are complete (or whenever the project owner would like)

**Release 0.1-alpha** (Date: 9/28/20)

This release focuses on providing a Web-UI to annotate sets of documents. As mentioned the user can create annotations for three tasks (NER, SA, RE), as well as provide explanations for those annotations in two different ways. We support various import/export formats, and take advantage of two recommendation strategies for NER (Noun phrase chunking--powered by spaCy, Historical Annotation Application) and one for RE (Historical Annotation Application). To further enable Historical Annotation Application, a project owner can upload a set of (x,y) pairs as "Historical" annotations, and we will apply these pairs as recommendations for un-annotated documents (sort of like distant learning).
 
# Installation Instructions:

Note: All paths are relative to being just outside the `LEAN-LIFE` directory. Please adjust paths accordingly.

* Please install [Python 3.6.5](https://www.python.org/downloads/release/python-365/) (if you use `conda` you can ignore this step)
* Please install [Postgres 12.3](http://postgresguide.com/setup/install.html) (in the linked example they use PostgreSQL 9.2, please ensure you replace 9.2 with 12.3)
* If on linux, please make sure to start postgres `sudo service postgresql start`
  - if you use the installation guide above for unix or windows you shouldn't have to do this
* Open a new terminal window after installing the above
* Clone this repo: `git clone git@github.com:INK-USC/LEAN-LIFE.git`
* Create a virtual environment using:
     * annaconda: `conda create -n leanlife python=3.6` (annaconda doesn't have a stable 3.6.5 version, so we use 3.6)
     * virtualenv:
          1. `python3.6.5 -m pip install virtualenv`
          2. `python3.6.5 -m venv leanlife`
* Activate your environment:
     * annaconda: `conda activate leanlife`
     * virtualenv: `source leanlife/bin/activate`
* `pip install -r requirements.txt` 
* `python -m spacy download en` (or whatever version you'd like, just make sure to update this in [utils.py](https://github.com/INK-USC/LEAN-LIFE/blob/master/annotation/src/server/utils.py#L8))
* Please install [NodeJS 14.16.0](https://nodejs.org/en/)

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
2. We will now setup the postgres connection. Navigate to the `src` folder inside `annotation`, `cd LEAN-LIFE/annotation/src`.
     * Inside the `app` folder, navigate to [settings.py](https://github.com/INK-USC/LEAN-LIFE/blob/master/annotation/src/app/settings.py#L100)
          * Find the `DATABASES` dictionary, and replace the `PASSWORD` "fill-this-in" with your own password
3. Navigate to the `src` folder inside `annotation`, `cd LEAN-LIFE/annotation/src` and run:
    * `./setup.sh PASSWORD-YOU-JUST-SET` <- (passing your password in as an argument)
    * you will be asked to create a user here, this user is what you will use to login to the LEAN-LIFE application
4. Start the backend: `python manage.py runserver 0.0.0.0:8000`
5. Navigate to the `frontend` folder inside `annotation/src`, `cd LEAN-LIFE/frontend`
6. We will now install required package for the frontend, `npm install`
7. Start the frontend: `npm run serve`
8. Open up an browser window and navigate to http://0.0.0.0:8080/

# Contributing

We love contributions, so thank you for taking the time! Pushing changes to master is blocked, so please create a branch and make your edits on the branch. Once done, please create a Pull Request and ask a contributer from the INK-LAB to pull your changes in. You can refer to our PR guidelines and general contribution guidelines [here](./CONTRIBUTING.md).

# Misc.

### Feedback
Feedback is definitely encouraged, please feel free to create an issue and document what you're seeing/wanting to see.

### Mailing List
To get notifications of major updates to this project, you can join our mailing list [here](https://groups.google.com/forum/#!forum/leanlife)

### Twitter
For updates on this project and other nlp projects being done at USC, please follow [@nlp_usc](https://twitter.com/nlp_usc)

### Contributors
Rahul Khanna, Dongho Lee, Jamin Chen, Seyeon Lee, JiaMin (Jim) Gong

We love contributions, so thank you for taking the time! Pushing changes to master is blocked, so please create a branch and make your edits on the branch. Once done, please create a Pull Request and ask a contributer from the INK-LAB to pull your changes in. You can refer to our PR guidelines and general contribution guidelines [here](./CONTRIBUTING.md).

# Misc.

### Feedback
Feedback is definitely encouraged, please feel free to create an issue and document what you're seeing/wanting to see.

### Mailing List
To get notifications of major updates to this project, you can join our mailing list [here](https://groups.google.com/forum/#!forum/leanlife)

### Twitter
For updates on this project and other nlp projects being done at USC, please follow [@nlp_usc](https://twitter.com/nlp_usc)

### Contributors
Rahul Khanna, Dongho Lee, Jamin Chen, Seyeon Lee, JiaMin (Jim) Gong


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
