import argparse
import json

from json_schema import ExplanationTrainingPayload
from main import start_next_training_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_path",
                        type=str,
                        default="./data/semeval_labeled.json",
                        help="Path to labled+explanation data.")
    parser.add_argument("--unlabeled_path",
                        type=str,
                        default="./data/semeval_unlabeled.json",
                        help="Path to unlabled data.")
    parser.add_argument("--dev_path",
                        type=str,
                        default="./data/semeval_dev.json",
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

    labeled_data = labeled_data[:10]
    unlabeled_data = unlabeled_data[:100]
    dev = dev[:50]

    data = {
        'params': {
            'experiment_name': 'semeval_strict_match',
            'dataset_name': 'semeval',
            "dataset_size": len(unlabeled_data),
            'task': 're',
            'build_data': True,
            "match_batch_size": 50,
            "unlabeled_batch_size": 100,
            "none_label_key": "no_relation",
            "learning_rate": 0.1,
            "embeddings": "glove.840B.300d",
            "hidden_dim": 100,
            # "random_state": 7698,
            "load_model": False,
            "start_epoch": 0,
            # "gamma": 0.5,
            "emb_dim": 300,
            "epochs": args.epochs,
            "eval_data": dev,
            "pre_train_hidden_dim": 300,
            "pre_train_training_size": 50000,
            "stage": "both",
            "pre_train_build_data": True,
            "soft_match": False,
        },
        'label_space': {
            "Product-Producer(e1,e2)": 0,
            "Component-Whole(e2,e1)": 1,
            "Product-Producer(e2,e1)": 2,
            "Instrument-Agency(e2,e1)": 3,
            "Message-Topic(e1,e2)": 4,
            "Entity-Origin(e2,e1)": 5,
            "Entity-Origin(e1,e2)": 6,
            "Entity-Destination(e1,e2)": 7,
            "Cause-Effect(e1,e2)": 8,
            "Content-Container(e1,e2)": 9,
            "Member-Collection(e2,e1)": 10,
            "Instrument-Agency(e1,e2)": 11,
            "Message-Topic(e2,e1)": 12,
            "Content-Container(e2,e1)": 13,
            "no_relation": 14,
            "Member-Collection(e1,e2)": 15,
            "Entity-Destination(e2,e1)": 16,
            "Component-Whole(e1,e2)": 17,
            "Cause-Effect(e2,e1)": 18
        },
        'explanation_triples': labeled_data,
        'unlabeled_text': unlabeled_data,
        "ner_label_space": [
            'ORGANIZATION',
            'PERSON',
            'MONEY',
            'NATIONALITY',
            'SET',
            'COUNTRY',
            'NUMBER',
            'MISC',
            'CAUSE_OF_DEATH',
            'STATE_OR_PROVINCE',
            'DATE',
            'IDEOLOGY',
            'ORDINAL',
            'RELIGION',
            'LOCATION',
            'TIME',
            'DURATION',
            'CRIMINAL_CHARGE',
            'TITLE',
            'O'
        ]
    }

    payload = ExplanationTrainingPayload(**data)
    data = start_next_training_api(payload)
    print(data)


if __name__ == "__main__":
    main()
