import argparse
import json

from json_schema import EvalStandardClfPayload
from main import eval_standard_clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",
                        type=str,
                        default="dummy",
                        help="Experiment name prefix.")
    parser.add_argument("--test_path",
                        type=str,
                        default="./data/semeval_test.json",
                        help="Path to train data.")

    args = parser.parse_args()

    test = json.load(open(args.test_path))
    test_formatted = list(map(lambda x: (x['text'], x['label']), test))

    # test_formatted = test_formatted[:1000]

    data = {
        'params': {
            'experiment_name': '%s_standard' % args.prefix,
            'dataset_name': 'semeval_%s' % args.prefix,
            'task': 'sa',
            "match_batch_size": 50,
            "none_label_key": "neutral",
            "embeddings": "glove.840B.300d",
            "hidden_dim": 100,
            "emb_dim": 300,
        },
        'eval_data': test_formatted,
        'label_space': {
            "positive": 0,
            "negative": 1,
            "neutral": 2,
        },
    }

    payload = EvalStandardClfPayload(**data)
    data = eval_standard_clf(payload)
    print(data)


if __name__ == "__main__":
    main()
