import argparse
import traceback
import os
import sys

from datetime import datetime
from morph_disamb.config import Config, NetworkTypes
from morph_disamb.data_handler import NeuralNetworkDataHandler
from morph_disamb.disambiguator import MorphologicalDisambiguator
from morph_disamb.hungarian_loader import HungarianDataLoader
from morph_disamb.neural_network import MorphologicalDisambiguationNeuralNetwork


def main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--recurrent', action='store_true',
                        help='Set neural network type to recurrent. -C or -R parameter is required.')
    parser.add_argument('-C', '--convolutional', action='store_true',
                        help='Set neural network type to convolutional. -C or -R parameter is required.')
    parser.add_argument('-l', '--load', action='store',
                        help='Loading neural network model. The parameter is the date and time in the name of the saved'
                             ' neural network which has to be used. Required format: yyyy-MM-dd-hh-mm.')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train the neural network.')
    parser.add_argument('-e', '--evaluate', action='store_true',
                        help='Evaluate the neural network.')
    parser.add_argument('-d', '--disambiguate', action='store_true',
                        help='Use the neural network for disambiguation.')
    parser.add_argument('-B', '--batch', type=int, action='store', default=64,
                        help='Batch size for the neural network.')
    parser.add_argument('-E', '--epoch', type=int, action='store', default=128,
                        help='Number of max epochs.')
    parser.add_argument('-D', '--directory', action='store',
                        help='Directory path of the disambiguation input file(s).')
    parser.add_argument('-F', '--file', action='store',
                        help='The input file which has to be processed. Only needed if not all files have to be '
                             'processed in the given directory.')
    parser.add_argument('--transducer', action='store', help='The transducer path.')
    args = parser.parse_args()

    success = change_config(args)
    if not success:
        parser.print_help()
        return
    run(args)


def main_params(network_type,
                load=None,
                train=False,
                evaluate=False,
                disambiguate=False,
                batch=64,
                epoch=128,
                directory=None,
                file=None,
                transducer=None):
    params = argparse.Namespace()
    params.recurrent = (network_type == NetworkTypes.RECURRENT)
    params.convolutional = (network_type == NetworkTypes.CONVOLUTIONAL)
    params.load = load
    params.train = train
    params.evaluate = evaluate
    params.disambiguate = disambiguate
    params.batch = batch
    params.epoch = epoch
    params.directory = directory
    params.file = file
    params.transducer = transducer

    success = change_config(params)
    if not success:
        return
    run(params)


def run(args):
    print(args)
    start_time = datetime.now()

    output_dir = os.path.join(Config.base_dir, 'output')
    create_not_existing_dirs([Config.save_dir, output_dir])

    loader = HungarianDataLoader()
    handler = NeuralNetworkDataHandler(loader)
    neural_network = MorphologicalDisambiguationNeuralNetwork(handler)

    # change root file path when loading a model and load roots
    if args.load is not None:
        root_file_name = neural_network.get_file_name(Config.network_type,
                                                      Config.neural_network_save_time) + '-roots.txt'
        Config.root_file = os.path.join(Config.save_dir, root_file_name)
        loader.load_root_vocabulary_from_file()

    # loading data
    loader.load_possible_word_analyses(Config.analyses_file)
    if args.train or args.evaluate:
        try:
            loader.load_input_from_file(Config.corpus_dir, Config.input_file)
            # saving roots
            if loader.new_root_added:
                loader.save_root_vocabulary_to_file()
        except Exception as e:
            print(e)
            traceback.print_exc()

    # loading or building model
    if args.load is not None:
        loaded = neural_network.load_model(Config.save_dir, Config.network_type, Config.neural_network_save_time)
        if not loaded:
            print('Error while loading neural network model.')
            return
    elif args.train or args.evaluate or args.disambiguate:
        neural_network.build_model()

    # training and evaluation data collecting
    if args.train or args.evaluate:
        handler.create_network_input_data()

    # training model and saving vocabulary
    if args.train:
        root_file_name = neural_network.get_file_name(neural_network.network_type,
                                                      neural_network.model_build_time) + '-roots.txt'
        Config.root_file = os.path.join(Config.save_dir, root_file_name)
        loader.save_root_vocabulary_to_file()

        neural_network.train_model()
        # load the best saved model
        neural_network.load_model(Config.save_dir, neural_network.network_type, neural_network.model_build_time)

    # evaluation: network & disambiguation
    if args.evaluate:
        try:
            neural_network.evaluate_model()
        except Exception as e:
            print(e)
            traceback.print_exc()
        result_file = os.path.join(output_dir, 'disambiguated-' + Config.neural_network_save_time + '.txt')
        disambiguator = MorphologicalDisambiguator(neural_network, result_file)
        disambiguator.disambiguate_sentences()

    # disambiguation
    if args.disambiguate:
        Config.training_percent = 0.0   # no shuffle
        disambiguator = MorphologicalDisambiguator(neural_network)

        if args.directory and args.file:
            # from file
            loader.load_input_from_file(args.directory, args.file)
            disambiguator.disambiguate_sentences(evaluation=False)

        # read text from standard input
        print("\n\033[94mWrite one or more sentences...\033[0m")
        for text in sys.stdin:
            loader.tokenize_input_text(text)
            disambiguator.disambiguate_sentences(evaluation=False)
            print("\n\033[94mWrite one or more sentences...\033[0m")

    end_time = datetime.now()
    print('\nStart time  : \033[94m', start_time, '\033[0m')
    print('End time    : \033[94m', end_time, '\033[0m')
    print('Time passed : \033[94m', end_time - start_time, '\033[0m')


def change_config(args):
    Config.batch_size = args.batch
    Config.epoch = args.epoch
    Config.to_evaluate = args.evaluate
    Config.to_disambiguate = args.disambiguate
    Config.transducer = args.transducer

    if args.directory is not None:
        Config.corpus_dir = args.directory

    if args.file is not None:
        Config.input_file = args.file

    # setting network type
    if args.recurrent:
        Config.network_type = NetworkTypes.RECURRENT
    elif args.convolutional:
        Config.network_type = NetworkTypes.CONVOLUTIONAL
    else:
        return False

    # don't change roots if it's not training
    if not args.train:
        Config.static_roots = True

    # saving the parameter network time
    if args.load is not None:
        Config.neural_network_save_time = args.load

    return True


def create_not_existing_dirs(directories):
    for dir in directories:
        os.makedirs(dir, exist_ok=True)


if __name__ == '__main__':
    main_args()
