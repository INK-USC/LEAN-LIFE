import argparse
import json

from json_schema import ExplanationTrainingPayload
from main import start_next_training_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",
                        type=str,
                        default="dummy",
                        help="Experiment name prefix.")
    parser.add_argument("--labeled_path",
                        type=str,
                        default="./data/sa_labeled.json",
                        help="Path to labled+explanation data.")
    parser.add_argument("--unlabeled_path",
                        type=str,
                        default="./data/sa_unlabeled.json",
                        help="Path to unlabled data.")
    parser.add_argument("--dev_path",
                        type=str,
                        default="./data/sa_dev.json",
                        help="Path to dev data.")
    parser.add_argument("--epochs",
                        default=75,
                        type=int,
                        help="Number of Epochs for training")
    parser.add_argument("--soft_match",
                        default="true",
                        type=str,
                        help="Boolean string")

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
            'experiment_name': '%s_%s_match' % (args.prefix, "soft" if args.soft_match.lower() == "true" else "strict"),
            'dataset_name': 'semeval_%s' % args.prefix,
            "dataset_size": len(unlabeled_data),
            'task': 'sa',
            'build_data': True,
            "match_batch_size": 50,
            "unlabeled_batch_size": 100,
            "none_label_key": "neutral",
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
            "soft_match": args.soft_match.lower() == "true"
        },
        'label_space': {
            "positive": 0,
            "negative": 1,
            "neutral": 2,
        },
        'explanation_triples': labeled_data,
        'unlabeled_text': unlabeled_data,
    }

    payload = ExplanationTrainingPayload(**data)
    data = start_next_training_api(payload)
    print(data)


if __name__ == "__main__":
    main()
