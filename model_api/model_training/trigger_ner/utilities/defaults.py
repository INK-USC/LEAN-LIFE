TRIG_MODEL_DEFAULTS = {
    "embeddings": "glove.6B.100d",
    "emb_dim": 100,
    "content_emb": "none",
    "hidden_dim": 200,
    "dropout": 0.5,
    "use_char_rnn": True,
    "use_crf_layer": True,
}

TRIG_TRAINING_DEFAULTS = {
    "seed": 42,
    "digit2zero": True,
    "optimizer": "sgd",
    "trig_optimizer": "adam",
    "learning_rate": 0.01,
    "momentum": 0.0,
    "l2": 1e-8,
    "num_epochs": 10,
    "pre_train_num_epochs": 20,
    "batch_size": 10,
    "lr_decay": 0,
}


class IOBES:
    B = "B-"
    I = "I-"
    S = "S-"
    E = "E-"
    O = "O"
