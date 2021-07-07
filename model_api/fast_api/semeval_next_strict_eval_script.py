import argparse
import json

from json_schema import EvalStandardClfPayload
from main import eval_standard_clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path",
                        type=str,
                        default="./data/semeval_test.json",
                        help="Path to train data.")

    args = parser.parse_args()

    test = json.load(open(args.test_path))
    test_formatted = list(map(lambda x: (x['text'], x['label']), test))

    test_formatted = test_formatted[:1000]

    data = {
        'params': {
            'experiment_name': 'semeval_strict_match',
            'dataset_name': 'semeval',
            'task': 're',
            "match_batch_size": 50,
            "none_label_key": "no_relation",
            "embeddings": "glove.840B.300d",
            "hidden_dim": 100,
            "emb_dim": 300,
        },
        'eval_data': test_formatted,
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

    payload = EvalStandardClfPayload(**data)
    data = eval_standard_clf(payload)
    print(data)


if __name__ == "__main__":
    main()
