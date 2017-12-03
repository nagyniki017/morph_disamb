from enum import Enum

import os


class NetworkTypes(Enum):
    RECURRENT = 0
    CONVOLUTIONAL = 1


class Config:
    window = 5
    training_percent = 0.9
    evaluation_percent = 0.9
    network_type = NetworkTypes.RECURRENT
    unknown_root = '<UNK>'
    sentence_separator = '<S>'
    number = '<NUM>'
    static_roots = False
    batch_size = 64
    epoch = 128
    neural_network_save_time = ''

    base_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'save')
    corpus_dir = ''
    analyses_file = os.path.join(base_dir, 'data', 'hfst_analyses.txt')
    features_file = os.path.join(base_dir, 'data', 'hungarian_features.txt')
    root_file = os.path.join(base_dir, 'save', 'roots.txt')
    input_file = None

    to_evaluate = False
    to_disambiguate = False

    transducer = ''

    lstm_optimizer = 'adam'
    lstm_loss = 'binary_crossentropy'
    lstm_activation = 'tanh'
    lstm_dropout = 0.25

    conv_optimizer = 'adagrad'
    conv_loss = 'categorical_crossentropy'
    conv_activation = 'relu'

    embedding_output_dim = 3
    patience = 5
