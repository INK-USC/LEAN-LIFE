import argparse
import json

from json_schema import EvalNextClfPayload
from main import eval_next_clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",
                        type=str,
                        default="dummy",
                        help="Experiment name prefix.")
    parser.add_argument("--unlabeled_path",
                        type=str,
                        default="./data/sa_unlabeled.json",
                        help="Path to unlabled data.")
    parser.add_argument("--test_path",
                        type=str,
                        default="./data/sa_test.json",
                        help="Path to train data.")
    parser.add_argument("--soft_match",
                        default="true",
                        type=str,
                        help="Boolean string")

    args = parser.parse_args()

    unlabeled_data = json.load(open(args.unlabeled_path))
    test = json.load(open(args.test_path))
    test_formatted = list(map(lambda x: (x['text'], x['label']), test))

    # unlabeled_data = unlabeled_data[:100]
    # test_formatted = test_formatted[:100]

    data = {
        'params': {
            'experiment_name': '%s_%s_match' % (args.prefix, "soft" if args.soft_match.lower()== "true" else "strict"),
            'dataset_name': 'semeval_%s' % args.prefix,
            "train_dataset_size": len(unlabeled_data),
            'task': 'sa',
            "eval_batch_size": 50,
            "none_label_key": "neutral",
            "embeddings": "glove.840B.300d",
            "hidden_dim": 100,
            "emb_dim": 300,
            "pre_train_training_size": 50000
        },
        'eval_data': test_formatted,
        'label_space': {
            "positive": 0,
            "negative": 1,
            "neutral": 2,
        },
    }

    payload = EvalNextClfPayload(**data)
    data = eval_next_clf(payload)
    print(data)


if __name__ == "__main__":
    main()
