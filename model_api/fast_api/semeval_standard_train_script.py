import argparse
import json

from json_schema import StandardPipelinePayload
from main import start_standard_training_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",
                        type=str,
                        default="./data/semeval_train.json",
                        help="Path to unlabled data.")
    parser.add_argument("--dev_path",
                        type=str,
                        default="./data/semeval_dev.json",
                        help="Path to dev data.")
    parser.add_argument("--test_path",
                        type=str,
                        default="./data/semeval_test.json",
                        help="Path to test data.")
    parser.add_argument("--epochs",
                        default=60,
                        type=int,
                        help="Number of Epochs for training")

    args = parser.parse_args()

    train = json.load(open(args.train_path))
    dev = json.load(open(args.dev_path))
    test = json.load(open(args.test_path))
    test = list(map(lambda x: (x['text'], x['label']), test))

    train = train[:100]
    dev = dev[:200]
    test = test[:10]

    data = {
        'params': {
            'experiment_name': 'semeval_standard',
            'dataset_name': 'semeval',
            'task': 're',
            'build_data': True,
            "match_batch_size": 50,
            "none_label_key": "no_relation",
            "learning_rate": 0.1,
            "embeddings": "glove.840B.300d",
            "hidden_dim": 100,
            "random_state": 7698,
            "load_model": False,
            "start_epoch": 0,
            "emb_dim": 300,
            "epochs": args.epochs,
            "eval_data": test
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
        'labeled_data': train,
        'dev_data': dev,
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

    payload = StandardPipelinePayload(**data)
    data = start_standard_training_api(payload)
    print(data)


if __name__ == "__main__":
    main()
