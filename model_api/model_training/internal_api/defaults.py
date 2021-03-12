FIND_MODULE_DEFAULTS : {
    "lower_bound" : -20.0,
    "pre_train_batch_size" : 64,
    "pre_train_eval_batch_size" : 128,
    "pre_train_learning_rate" : 0.001,
    "pre_train_epochs" : 20,
    "embeddings" : "glove.840B.300d",
    "pre_train_gamma" : 0.5,
    "pre_train_emb_dim" : 300,
    "pre_train_hidden_dim" : 300,
    "pre_train_training_size" : 50000,
    "pos_weight" : 20.0,
    "clip_gradients" : 1.0,
    "pre_train_random_state" : 42
}

BILSTM_DEFAULTS : {
    
}

TRAINING_DEFAULTS : {
    "load_model" : False,
    "start_epoch" : 0
}
