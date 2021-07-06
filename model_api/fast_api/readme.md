Implementation details about NeXT Framework can be found [here](../model_training/next_framework/readme.md)

**Important Notes while training:**

1. For RE task, the entity type matching is relaxed while applying soft labeling functions. To ensure type matching occurs, `relation_ner_types` needs to populated using one of the below methods -
   1. _**Passing in request body**_ (recommended): you will have to pass `relation_ner_types` in the request body while placing train, predict or evaluate API call.
   2. _**Defining in code**_: define it the code like done for tacred at these locations: [training](../model_training/internal_api/next_pipelines.py#L425), [evaluation](../model_training/internal_api/next_pipelines.py#L987) and [prediction](../model_training/internal_api/next_pipelines.py#L1088).

# Table of contents

1. [Steps to run the FAST api](#steps-to-run-the-fast-api)
2. [Schema Definition](#schema-definition)

   2.1. [Standard Training API](#1-standard-training-api)

   2.2. [NeXT Prediction API](#2-standard-prediction-api)

   2.3. [NeXT Evaluation API](#3-standard-evaluation-api)

   2.4. [NeXT Training API](#4-next-training-api)

   2.5. [NeXT Prediction API](#5-next-prediction-api)

   2.6. [NeXT Evaluation API](#6-next-evaluation-api)

# Steps to run the FAST api

1. Ensure your `leanlife` environment is activated
2. Execute `sh create_data_dirs.sh`
3. Execute `pip install -r rqs.txt`
4. modify nltk source code: in `nltk/parse/chart.py`, line 680, modify function `parse`,
   change `for edge in self.select(start=0, end=self._num_leaves,lhs=root):`
   to `for edge in self.select(start=0, end=self._num_leaves):`
5. `python -m spacy download en_core_web_sm`
6. `uvicorn main:app --reload --port=9000`
7. The API is now exposed on port 9000. Swagger interactive API documentation can be
   found [here](http://127.0.0.1:9000/docs)

**CUDA**:

Model Training Code will require CUDA 10.1, and if you're on linux you'll have to install it via the pytorch site. The
torch version is 1.7.0. For more help here, check [here](https://pytorch.org/get-started/previous-versions/#v170)

Additionally to set the GPUs available for model training, please change line 28 in `main.py`

# Schema Definition

The complete schema definition can be found in [JSON schema file](json_schema.py). And can also be accessed via Swagger
interactive API documentation.

## 1. Standard Training API

- Route - `POST /training/standard/api/`
- Sample body -

```json
{
  "params": {
    "experiment_name": "imdb_sa_nle_standard",
    "dataset_name": "imdb_sa_nle_standard",
    "task": "sa",
    "match_batch_size": 50,
    "learning_rate": 0.1,
    "epochs": 5,
    "embeddings": "charngram.100d",
    "emb_dim": 100,
    "hidden_dim": 100,
    "random_state": 7698,
    "build_data": true
  },
  "label_space": {
    "Positive": 0,
    "Negative": 1
  },
  "labeled_data": [
    {
      "text": "Walt Disney's CINDERELLA takes a story everybody's familiar with and embellishes it with humor and suspense, while retaining the tale's essential charm. Disney's artists provide the film with an appealing storybook look that emanates delectable fairy tale atmosphere. It is beautifully, if conventionally, animated; the highlight being the captivating scene where the Fairy Godmother transforms a pumpkin into a majestic coach and Cinderella's rags to a gorgeous gown. Mack David, Al Hoffman, and Jerry Livingston provide lovely songs like A Dream Is a Wish Your Heart Makes\" and \"Bibbidi-Bobbidi-Boo\" that enhance both the scenario and the characters.<br /><br />Even though CINDERELLA's story is predictable",
      "label": "Positive"
    },
    {
      "text": "I think this is one of the weakest of the Kenneth Branagh Shakespearian works. After such great efforts as Much Ado About Nothing, etc. I thought this was poor. The cast was weaker (Alicia Silverstone, Nivoli, McElhone???) but my biggest gripe was that they messed with the Bard's work and cut out some of the play to put in the musical/dance sequences.<br /><br />You just don't do Shakespeare and then mess with the play. Sorry, but that is just wrong. I love some Cole Porter just like the next person, but jeez, don't mess with the Shakespeare. Skip this and watch Prospero's Books\" if you want to see a brilliant Shakespearean adaptation of the Tempest.\"",
      "label": "Negative"
    }
  ]
}
```

- Sample Response -

```json
{
  "save_path": "/home/Projects/ink-lab-lean-life/LEAN-LIFE/model_api/fast_api/../model_training/internal_api/../next_framework/data/saved_models/Clf_imdb_sa_nle_standard.p"
}
```

---

## 2. Standard Prediction API

- Route - `POST /training/standard/predict`
- Sample body -

```json
{
  "params": {
    "experiment_name": "imdb_sa_nle_standard",
    "dataset_name": "imdb_sa_nle_standard",
    "task": "sa",
    "eval_batch_size": 50,
    "embeddings": "charngram.100d",
    "emb_dim": 100,
    "hidden_dim": 100
  },
  "label_space": {
    "Positive": 0,
    "Negative": 1
  },
  "prediction_data": [
    "This film is underrated. I loved it. It was truly sweet and heartfelt. A family who struggles but isn't made into a dysfunctional family which is so typical of films today. The film didn't make it an issue that they have little money or are Dominican Republican the way Hollywood have.<br /><br />Instead the issue is Victor is immature and needs to grow up. He does, slowly, by the film's end. He has a ways to go, but it was a heartfelt attempt to move forward. His grandmother is very cute and the scene where the little boy throws up had me laughing for the longest time. A truly heartfelt indie"
  ]
}
```

- Sample Response -

```json
{
  "class_probs": [[0.48354560136795044, 0.5164543986320496]],
  "class_preds": ["Negative"]
}
```

---

## 3. Standard Evaluation API

- Route - `POST /training/standard/eval`
- Sample body -

```json
{
  "params": {
    "experiment_name": "imdb_sa_nle_standard",
    "dataset_name": "imdb_sa_nle_standard",
    "task": "sa",
    "eval_batch_size": 50,
    "embeddings": "charngram.100d",
    "emb_dim": 100,
    "hidden_dim": 100
  },
  "label_space": {
    "Positive": 0,
    "Negative": 1
  },
  "eval_data": [
    [
      "This film is underrated. I loved it. It was truly sweet and heartfelt. A family who struggles but isn't made into a dysfunctional family which is so typical of films today. The film didn't make it an issue that they have little money or are Dominican Republican the way Hollywood have.<br /><br />Instead the issue is Victor is immature and needs to grow up. He does, slowly, by the film's end. He has a ways to go, but it was a heartfelt attempt to move forward. His grandmother is very cute and the scene where the little boy throws up had me laughing for the longest time. A truly heartfelt indie",
      "Positive"
    ],
    [
      "I never thought I would absolutly hate an Arnold Schwartzeneggar film, BUT this is is dreadful from the get go. there isnt one redeemable scene in the entire 123 long minutes. an absolute waste of time<br /><br /> thank yu<br /><br /> Jay harris",
      "Negative"
    ]
  ]
}
```

- Sample Response -

```json
{
  "avg_loss": 0.68117,
  "avg_eval_f1_score": 0.7
}
```

---

## 4. NeXT Training API

- Route - `POST /training/next/api/`
- Sample body -

```json
{
  "params": {
    "experiment_name": "imdb_sa_nle_soft_match",
    "dataset_name": "imdb_sa_nle_soft_match",
    "dataset_size": 94,
    "task": "sa",
    "match_batch_size": 50,
    "unlabeled_batch_size": 100,
    "learning_rate": 0.1,
    "epochs": 5,
    "embeddings": "charngram.100d",
    "emb_dim": 100,
    "gamma": 0.5,
    "hidden_dim": 100,
    "random_state": 7698,
    "pre_train_hidden_dim": 300,
    "pre_train_training_size": 50000,
    "soft_match": true,
    "stage": "both",
    "pre_train_build_data": true,
    "build_data": true
  },
  "label_space": {
    "Positive": 0,
    "Negative": 1
  },
  "explanation_triples": [
    {
      "text": "Walt Disney's CINDERELLA takes a story everybody's familiar with and embellishes it with humor and suspense, while retaining the tale's essential charm. Disney's artists provide the film with an appealing storybook look that emanates delectable fairy tale atmosphere. It is beautifully, if conventionally, animated; the highlight being the captivating scene where the Fairy Godmother transforms a pumpkin into a majestic coach and Cinderella's rags to a gorgeous gown. Mack David, Al Hoffman, and Jerry Livingston provide lovely songs like A Dream Is a Wish Your Heart Makes\" and \"Bibbidi-Bobbidi-Boo\" that enhance both the scenario and the characters.<br /><br />Even though CINDERELLA's story is predictable",
      "label": "Positive",
      "explanation": "The word 'embellishes' appears in the text"
    },
    {
      "text": "You want to know what the writers of this movie consider funny? A robot child sees his robot parents killed (beheaded, as I recall), and then moves between their bodies calling their names. Yeah--what a comic moment. This is the worst movie I ever paid to see.",
      "label": "Negative",
      "explanation": "The word 'worst' appears in the text"
    }
  ],
  "unlabeled_text": [
    "The idea of making a film about the Beatles sounds doomed idea, as no production can catch the idea of the actual historic Beatles. Then it is perhaps best not to try to recreate the past, but to produce an illustration that works best with the other available Beatles material. This is exactly what 'Birth of the Beatles' offers to us, the simple story known to us without any extravaganza.<br /><br />*** SPOILERS here on *** <br /><br />Be warned that not everything is that accurate as some Beatles-graduates might expect. The Beatles are seen performing songs that hardly were even composed by that time. The Beatles perform Ask Me Why\"",
    "There's a certain irony in a parody of the Gothic genre being turned into a mess of clichés by filmmakers who either had no idea what the story's purpose was, or just didn't care. All of the hallmarks of your average family film are present- rambunctious younger siblings, a grumpy teenager who doesn't want to move, unsympathetic parents who are unable to see the apparition, and of course a romantic subplot. The movie has very little in common with Wilde's original story, which was largely written to poke fun at the melodramatic Gothic novellas that were all the rage at the time. If Wilde saw this version, he'd probably laugh- and then of course, write a parody. One can only hope that the children who watched this bland, mass-produced pap eventually discovered the wit and sparkle of the original version."
  ]
}
```

- Sample Response -

```json
{
  "save_path": "/home/Projects/ink-lab-lean-life/LEAN-LIFE/model_api/fast_api/../model_training/internal_api/../next_framework/data/saved_models/Next-Clf_imdb_sa_nle_soft_match.p"
}
```

---

## 5. NeXT Prediction API

- Route - `POST /training/next/api/`
- Sample body -

```json
{
  "params": {
    "experiment_name": "imdb_sa_nle_soft_match",
    "dataset_name": "imdb_sa_nle_soft_match",
    "train_dataset_size": 94,
    "task": "sa",
    "eval_batch_size": 50,
    "embeddings": "charngram.100d",
    "emb_dim": 100,
    "hidden_dim": 100,
    "pre_train_training_size": 50000
  },
  "label_space": {
    "Positive": 0,
    "Negative": 1
  },
  "prediction_data": [
    "This film is underrated. I loved it. It was truly sweet and heartfelt. A family who struggles but isn't made into a dysfunctional family which is so typical of films today. The film didn't make it an issue that they have little money or are Dominican Republican the way Hollywood have.<br /><br />Instead the issue is Victor is immature and needs to grow up. He does, slowly, by the film's end. He has a ways to go, but it was a heartfelt attempt to move forward. His grandmother is very cute and the scene where the little boy throws up had me laughing for the longest time. A truly heartfelt indie"
  ]
}
```

- Sample Response -

```json
{
  "class_probs": [[0.3555419445037842, 0.6444580554962158]],
  "class_preds": ["Negative"]
}
```

---

## 6. NeXT Evaluation API

- Route - `POST /training/next/api/`
- Sample body -

```json
{
  "params": {
    "experiment_name": "imdb_sa_nle_soft_match",
    "dataset_name": "imdb_sa_nle_soft_match",
    "train_dataset_size": 94,
    "task": "sa",
    "eval_batch_size": 50,
    "embeddings": "charngram.100d",
    "emb_dim": 100,
    "hidden_dim": 100,
    "pre_train_training_size": 50000
  },
  "label_space": {
    "Positive": 0,
    "Negative": 1
  },
  "eval_data": [
    [
      "This film is underrated. I loved it. It was truly sweet and heartfelt. A family who struggles but isn't made into a dysfunctional family which is so typical of films today. The film didn't make it an issue that they have little money or are Dominican Republican the way Hollywood have.<br /><br />Instead the issue is Victor is immature and needs to grow up. He does, slowly, by the film's end. He has a ways to go, but it was a heartfelt attempt to move forward. His grandmother is very cute and the scene where the little boy throws up had me laughing for the longest time. A truly heartfelt indie",
      "Positive"
    ],
    [
      "Honestly, this is a very funny movie if you are looking for bad acting (Heather Graham could never live this down... it has three titles for a reason- to protect the guilty!), beautifully bad dialog (\"Do you like... ribs?\"), and a plot only a mother could approve, this is your Friday night entertainment! <br /><br />My roommate rented this under the title \"Terrified\" because he liked Heather Graham, but terrified is what we felt after the final credits. Not because the movie is scary, but because somebody actually paid money to make this turd on a movie reel.<br /><br />Horrible movie. There are a few no-name actors that provide some unintentional comedy, but nothing worth viewing. Heather Graham's dramatic climax also was one of the most pathetic and disturbing things I have ever witnessed. I award this movie no point, and may God have mercy on its soul.",
      "Negative"
    ]
  ]
}
```

- Sample Response -

```json
{
  "avg_loss": 0.61921,
  "avg_eval_ent_f1_score": 0.0,
  "avg_eval_val_f1_score": 0.7,
  "no_relation_thresholds": [0.0, -1.0]
}
```

---
