import argparse
import json

from json_schema import StandardPipelinePayload
from main import start_standard_training_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",
                        type=str,
                        default="dummy",
                        help="Experiment name prefix.")
    parser.add_argument("--train_path",
                        type=str,
                        default="./data/sa_train.json",
                        help="Path to unlabled data.")
    parser.add_argument("--dev_path",
                        type=str,
                        default="./data/sa_dev.json",
                        help="Path to dev data.")
    parser.add_argument("--test_path",
                        type=str,
                        default="./data/sa_test.json",
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

    # train = train[:100]
    # dev = dev[:200]
    # test = test[:10]

    data = {
        'params': {
            'experiment_name': '%s_standard' % args.prefix,
            'dataset_name': 'semeval_%s' % args.prefix,
            'task': 'sa',
            'build_data': True,
            "match_batch_size": 50,
            "none_label_key": "neutral",
            "learning_rate": 0.1,
            "embeddings": "glove.840B.300d",
            "hidden_dim": 100,
            # "random_state": 7698,
            "load_model": False,
            "start_epoch": 0,
            "emb_dim": 300,
            "epochs": args.epochs,
            "eval_data": test
        },
        'label_space': {
            "positive": 0,
            "negative": 1,
            "neutral": 2,
        },
        'labeled_data': train,
        'dev_data': dev,
    }

    payload = StandardPipelinePayload(**data)
    data = start_standard_training_api(payload)
    print(data)


if __name__ == "__main__":
    main()
