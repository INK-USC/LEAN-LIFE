import argparse
import json

from json_schema import ExplanationTrainingPayload
from main import start_next_training_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_path",
                        type=str,
                        default="./data/tacred_labeled.json",
                        help="Path to labled+explanation data.")
    parser.add_argument("--unlabeled_path",
                        type=str,
                        default="./data/tacred_unlabeled.json",
                        help="Path to unlabled data.")
    parser.add_argument("--dev_path",
                        type=str,
                        default="./data/tacred_dev.json",
                        help="Path to dev data.")
    parser.add_argument("--epochs",
                        default=60,
                        type=int,
                        help="Number of Epochs for training")

    args = parser.parse_args()

    labeled_data = json.load(open(args.labeled_path))
    unlabeled_data = json.load(open(args.unlabeled_path))
    dev = json.load(open(args.dev_path))
    dev = list(map(lambda x: (x['text'], x['label']), dev))

    # labeled_data = labeled_data[:10]
    # unlabeled_data = unlabeled_data[:100]
    # dev = dev[:50]

    data = {
        'params': {
            'experiment_name': 'tacred_soft_match',
            'dataset_name': 'tacred',
            "dataset_size": len(unlabeled_data),
            'task': 're',
            'build_data': True,
            "match_batch_size": 50,
            "unlabeled_batch_size": 100,
            "none_label_key": "no_relation",
            "learning_rate": 0.1,
            "embeddings": "glove.840B.300d",
            "hidden_dim": 100,
            "random_state": 7698,
            "load_model": False,
            "start_epoch": 0,
            "gamma": 0.5,
            "emb_dim": 300,
            "epochs": args.epochs,
            "eval_data": dev,
            "pre_train_hidden_dim": 300,
            "pre_train_training_size": 50000,
            "stage": "both",
            "pre_train_build_data": True
        },
        'label_space': {
            'org:subsidiaries': 0,
            'org:stateorprovince_of_headquarters': 1,
            'per:country_of_birth': 2,
            'per:charges': 3,
            'org:top_members/employees': 4,
            'per:stateorprovince_of_birth': 5,
            'org:members': 6,
            'per:country_of_death': 7,
            'org:alternate_names': 8,
            'per:siblings': 9,
            'per:other_family': 10,
            'org:country_of_headquarters': 11,
            'per:date_of_death': 12,
            'per:countries_of_residence': 13,
            'no_relation': 14,
            'per:city_of_birth': 15,
            'org:founded': 16,
            'org:website': 17,
            'org:member_of': 18,
            'per:alternate_names': 19,
            'org:shareholders': 20,
            'per:schools_attended': 21,
            'per:cause_of_death': 22,
            'per:date_of_birth': 23,
            'per:age': 24,
            'per:city_of_death': 25,
            'org:number_of_employees/members': 26,
            'per:stateorprovinces_of_residence': 27,
            'per:employee_of': 28,
            'per:stateorprovince_of_death': 29,
            'per:parents': 30,
            'per:children': 31,
            'per:title': 32,
            'per:religion': 33,
            'org:dissolved': 34,
            'org:parents': 35,
            'org:political/religious_affiliation': 36,
            'per:spouse': 37,
            'per:cities_of_residence': 38,
            'org:city_of_headquarters': 39,
            'org:founded_by': 40,
            'per:origin': 41
        },
        'explanation_triples': labeled_data,
        'unlabeled_text': unlabeled_data,
        "ner_label_space": [
            "PERSON",
            "ORGANIZATION",
            "DATE",
            "NUMBER",
            "TITLE",
            "COUNTRY",
            "LOCATION",
            "CITY",
            "MISC",
            "STATE_OR_PROVINCE",
            "DURATION",
            "NATIONALITY",
            "CAUSE_OF_DEATH",
            "CRIMINAL_CHARGE",
            "RELIGION",
            "URL",
            "IDEOLOGY"
        ]
    }

    payload = ExplanationTrainingPayload(**data)
    data = start_next_training_api(payload)
    print(data)


if __name__ == "__main__":
    main()
