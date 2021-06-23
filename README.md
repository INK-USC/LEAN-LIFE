# LEAN-LIFE

**L**abel **E**fficient **A**nnotatio**N** framework that allows for **L**earn**I**ng **F**rom **E**xplanations

[Website](http://inklab.usc.edu/leanlife/) &nbsp;&nbsp;&nbsp;&nbsp;
[Paper](https://arxiv.org/abs/2004.07499) &nbsp;&nbsp;&nbsp;&nbsp;

---

## Latest release (v1.0) Highlights

- Added new UI for annotation.
- FastAPI added for asynchronous model training.
- Added model training support for Natural Language Explanations for Relation Extraction and Sentiment Analysis tasks.

---

### Contents:

- [Quick Intro](#quick-intro)
- [Release Plan](#release-plan)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Misc](#misc)
- [Citation](#citation)

# Quick Intro:

LEAN-LIFE is an annotation framework for Named Entity Recognition (NER), Relation Extraction (RE) and Sentiment Analysis (SA)/Multi-Class Document Classification. LEAN-LIFE additionally enables the capture and use of explanations during the annotation process. Explanations can be seen as enhanced supervision, providing reasoning behind the labeling decision, and thus help speed up model training in low-resource environments.

Our initial frontend code is based on the [Doccano](https://github.com/chakki-works/doccano) project and the [AlpacaTag](https://github.com/INK-USC/AlpacaTag) project however we differentiate ourselves in these ways:

- **Triplet Capture**: Allows the building of a dataset that is (datapoint, label, explanation), unlike the standard (datapoint, label) tuple.

- **Explanation Supported Model Training**: Taining of both [TriggerNER](https://github.com/INK-USC/TriggerNER) and [NExT](https://github.com/INK-USC/NExT) models, for both model deployment and recommendations (coming soon)

- **Relation Extraction Task Supported**: Using the output of the Named Entity Extraction task, our system allows for the creation of relation extraction datasets.

Due to refactoring efforts and a desire to create a more stable framework the following features are not supported yet, but will be supported soon:

- **Active intelligent recommendation**: Iteratively train an appropriate backend model for the selected task using explanations and labels to both provide enhanced annotations to annotators and ensure annotators are not asked to provide annotations on documents that the model already understands.

- **Real-time model extraction**: Users can extract the trained recommendation model at any point for deployment purposes, without having to wait till all documents are labeled.

- **Model Interaction API**: Seperate API for model training (batch), prediction(online and batch), and extraction--this functionality will be built seperately from our annotation framework

- **User Roles**: Differentiating between a project creator and project annotators, allowing for a creator to set up a project, while allowing annotators to configure more local settings like what types of recommendations they would like, and how often their backend model should be trained.

Reference our [website](http://inklab.usc.edu/leanlife/) for more information.

For information on how to use the annotation framework and supported data formats, please look at our [wiki](https://github.com/INK-USC/LEAN-LIFE/wiki)

We strongly encourage community engagement, please refer to our [contribution section](#contributing) for more on how to contribute!

# Release Plan:

**Next Release's Goals:** 4-6 Weeks

- **NER training**: Include support training of NER task, using triggerNER.
- **Import & Export**: Add support for importing and exporting of labelled with or without explanations.
- **Docker**: Add docker files for easier onboarding/setup
- **Code cleanup** - remove unwanted files of django templates and add more inline documentation/explanations
- Improve CUDA management policy. Add support for per request modification of gpu device. Currently, set using environment variable.

Release notes can found [here](RELEASE.md).

# Getting Started:

<details>
<summary><b>Setup Instructions</b></summary>
<p>
Note: All paths are relative to being just outside the `LEAN-LIFE` directory. Please adjust paths accordingly.

1. Please install [Python 3.6.5](https://www.python.org/downloads/release/python-365/) (if you use `conda` you can ignore this step)
2. Open a new terminal window after installing the above
3. Clone this repo: `git clone git@github.com:INK-USC/LEAN-LIFE.git`
4. Create a virtual environment using:

   - annaconda: `conda create -n leanlife python=3.6` (annaconda doesn't have a stable 3.6.5 version, so we use 3.6)
   - virtualenv:
     1. `python3.6.5 -m pip install virtualenv`
     2. `python3.6.5 -m venv leanlife`

5. Activate your environment:

   - annaconda: `conda activate leanlife`
   - virtualenv: `source leanlife/bin/activate`

6. If you are user who wants to interact with the system using web portal, then please setup Django (for annotation backend) and Vue.js (for frontend). Otherwise, skip to the next step for model training backend setup.

   - Follow Django annotation backend installation instructions [here](annotation_backend/src/README.md)
   - Follow Vue.js frontend installation instructions [here](frontend/README.md)

7. Follow FastAPI/Model training installation instructions [here](model_api/fast_api/readme.md).
   For detailed guide on how to use API calls. Checkout the jupyter notebooks in [example_notebooks](example_notebooks/) directory

</p>
</details>
<br/>

<details>
<summary><b>Potential Errors</b></summary>
<p>

- Wrong version of python is being used.
  - To check: if you're getting installation errors, it could be that your machine is running the wrong version of python and/or installed packages. To check run `which python` and make sure the returned folder is the path to the `leanlife` virtual environment folder. To check that python is looking in the right places check this example [here](https://bic-berkeley.github.io/psych-214-fall-2016/sys_path.html#python-looks-for-modules-in-sys-path). Again the path should be the site-packages folder in your `leanlife` virtual environment
  - To Fix: Re-create virtual environment: - `deactivate leanlife` - `rm -rf leanlife` - make sure no other virtualenvs are running - open up terminal/command prompt and see if there are paranthesis at the start of each line, ex: `(base) user@...` - if this is the case deactivate that environment: `deactivate environment-name`, in the above example it would be `deactivate base` - Go to step 4 of installation instructions

</p>
</details>
<br/>

<details>
<summary><b>Directory overview</b></summary>
<p>

- `annotation_backend/`
  - `src/` - django applications root directory
  - `test_data/` - sample datasets for testing more description [here](annotation_backend/test_data/readme.md)
- `example_notebooks` -
  - Contains interactive jupyter notebooks for each tasks with sample data, processing and explanation to various input parameters.
  - execute `jupyter notebook` in terminal and access them
- `frontend/` - Vue.js frontend project directory
- `model_api/` - consists of code related to model training and fast api routes. More details can be found [here](model_api/README.md)
</p>
</details>
<br/>

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

Rahul Khanna, JiaMin (Jim) Gong, Dongho Lee, Jamin Chen, Seyeon Lee, Akshen Kadakia, Raghavendra Vedula

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
