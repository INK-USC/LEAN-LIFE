FIND_MODULE_DEFAULTS = {
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

BILSTM_DEFAULTS = {
    "clip_gradient_size" : 5.0,
    "layer_x_directions" : 4,
    "match_batch_size" : 50, 
    "unlabeled_batch_size" : 100,
    "learning_rate" : 0.1,
    "epochs" : 75,
    "gamma" : 0.7,
    "random_state" : 42,
    "embeddings" : "glove.840B.300d",
    "emb_dim" : 300,
    "hidden_dim" : 100,
    "none_label_key" : None,
    "eval_batch_size" : 100  
}

TRAINING_DEFAULTS = {
    "load_model" : False,
    "start_epoch" : 0
}
