import os
import sys
import traceback

from keras.layers import Embedding, Merge, LSTM, Dropout, Flatten, Dense, Conv1D
from keras.models import Sequential, model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from datetime import datetime

from morph_disamb.config import Config, NetworkTypes


class MorphologicalDisambiguationNeuralNetwork:
    __slots__ = ('data_handler', 'model', 'network_type', 'save_model_dir', 'model_build_time')

    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.model = Sequential()
        self.network_type = None
        self.model_build_time = None

        if os.path.isdir(Config.save_dir):
            self.save_model_dir = Config.save_dir
        else:
            self.save_model_dir = None
            print("\033[91mThe given parameter is not an existing directory!\033[0m")

    def build_model(self):
        """
        Builds a new model based on the neural network type and saves its building time.
        :return:
        """
        self.model_build_time = datetime.now()
        self.network_type = Config.network_type

        if self.network_type == NetworkTypes.RECURRENT:
            self.__build_recurrent_model()
        elif self.network_type == NetworkTypes.CONVOLUTIONAL:
            self.__build_convolutional_model()
        else:
            print("\n\033[91mWrong model type found in the configuration, can't build neural network!\033[0m")
            self.network_type = None

        self.model.summary()
        self.__compile_model()

    def load_model(self, load_dir, network_type, date_time_string):
        """
        Loads a neural network model and weights from the given folder. The name of the searched files include their
        type, a datetime string and a differing ending with the next format:
        <recurr|conv>-<datetime>-<model.json|weights.hd5>
        :param load_dir: The directory where the files can be found.
        :param network_type: The type of the network which has to be loaded.
        :param date_time_string: The datetime string which is part of the filenames.
        :return: Whether the loading was successful or not.
        """
        print('\n\033[92mLoading model from file...\033[0m')
        file_name = self.get_file_name(network_type, date_time_string)
        load_model_file = os.path.join(load_dir, file_name + '-model.json')
        load_weights_file = os.path.join(load_dir, file_name + '-weights.h5')
        print(load_model_file)
        print(load_weights_file)

        # only load files if both exist
        if os.path.isfile(load_model_file) and os.path.isfile(load_weights_file):
            self.network_type = network_type
            with open(load_model_file, mode='r') as model_file:
                json_string = model_file.read()
                self.model = model_from_json(json_string)
            self.__compile_model()
            self.model.load_weights(load_weights_file)

            self.model_build_time = datetime.now()
            return True
        return False

    def train_model(self):
        """
        Trains the built neural network model. It adds early stopping to the training, and saves the best weights to
        file. After the training it saves the model to file.
        :return:
        """
        self.__save_model_configuration()
        self.save_model(only_model=True)

        labels = self.data_handler.training_labels if self.network_type == NetworkTypes.RECURRENT else \
            np_utils.to_categorical(self.data_handler.training_labels, 2)

        weights_file = os.path.join(Config.save_dir, self.get_file_name() + '-weights.h5')
        # callbacks
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=Config.patience,
                                       verbose=1,
                                       mode='auto')
        checkpoint = ModelCheckpoint(weights_file,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True)
        callbacks = [checkpoint, early_stopping]

        print('\n\033[92mTraining model...\033[0m')
        # training, then saving the model and weights
        try:
            self.model.fit(self.data_handler.get_training_array(),
                           labels,
                           epochs=Config.epoch,
                           batch_size=Config.batch_size,
                           shuffle=False,
                           validation_split=1-Config.training_percent,
                           callbacks=callbacks,
                           verbose=1)
        except Exception as e:
            print('Error occurred:')
            traceback.print_exc()
            print(e)
        finally:
            self.save_model()

    def evaluate_model(self):
        """
        Evaluates the model on unseen data, and writes the achieved loss and accuracy to output.
        :return: The result of the evaluation.
        """
        print('\n\033[92mEvaluating model...\033[0m')
        labels = self.data_handler.evaluation_labels if self.network_type == NetworkTypes.RECURRENT else \
            np_utils.to_categorical(self.data_handler.evaluation_labels, 2)
        evaluation = self.model.evaluate(self.data_handler.get_evaluation_array(),
                                         labels,
                                         batch_size=Config.batch_size,
                                         verbose=1)

        print("Loss     :\033[91m", evaluation[0], '\033[0m')
        print("Accuracy :\033[91m", evaluation[1]*100, '\033[0m')
        self.save_evaluation(evaluation)
        return evaluation

    def save_model(self, only_model=False):
        """
        Saves the model into file, and also saves the weight into a different file if there was no such file created
        befoore. Earlier a model checkpoint could have been created with the best achieved results. If it exists, it
        won't be overwritten.
        :param only_model: Whether only the model should be saved or not.
        :return:
        """
        if len(self.save_model_dir) is not None and self.network_type is not None:
            print('\n\033[92mSaving model and weights...\033[0m')
            # saving model
            file_name = self.get_file_name()
            file_path = os.path.join(self.save_model_dir, file_name + '-model.json')
            print('Model file path   :\033[95m', file_path, '\033[0m')

            json_string = self.model.to_json()
            with open(file_path, mode='w') as json_save:
                json_save.write(json_string)

            # saving weights
            file_path = os.path.join(self.save_model_dir, file_name + '-weights.h5')

            if os.path.isfile(file_path):
                print('Weights file path :\033[95m', file_path, '\033[0m')
                print("Weights file already exists and won't be overridden")
            elif not only_model:
                print('Weights file path :\033[95m', file_path, '\033[0m')
                self.model.save_weights(file_path)
        else:
            print("\n\033[92mCan't save model, directory or network type is unknown...\033[0m")

    def predict_data(self, sentence_data):
        """
        Feeds the input data into the neural network for prediction and returns the results.
        :param sentence_data: The inputs (which are usually created from a sentence).
        :return: The result of the prediction in a list.
        """
        return self.model.predict(sentence_data)

    def save_evaluation(self, evaluation):
        file_name = self.get_file_name()
        file_path = os.path.join(self.save_model_dir, file_name + '-log.txt')
        mode = 'a' if os.path.isfile(file_path) else 'w'

        with open(file_path, mode=mode) as f:
            f.write("\nLoss:\t" + str(evaluation[0]))
            f.write("\nAccuracy:\t" + str(evaluation[1] * 100) + '\n')

    def get_file_name(self, network_type=None, date_time=None):
        """
        Returns a file name created from the type of the network and a datetime information.
        The returned format is <network type>[-<datetime>] where the datetime is optional.
        :param network_type: The type of the network.
        :param date_time: The datetime to put into the filename as string.
        :return: The created filename string without extension.
        """
        if network_type is None:
            network_type = self.network_type
        if date_time is None:
            date_time = self.__get_filename_time()

        if network_type == NetworkTypes.RECURRENT:
            file_name = 'recurr'
        else:
            file_name = 'conv'

        if isinstance(date_time, datetime):
            return file_name + '-' +str(date_time.strftime("%Y-%m-%d-%H-%M"))
        return file_name + '-' + date_time

    def __build_recurrent_model(self):
        """
        Builds a recurrent model using LSTM layers, saves its configuration into file and compiles the model.
        :return:
        """
        print('\n\033[92mBuilding recurrent model...\033[0m')
        embedding_models = self.__get_embeddigs()

        self.model = Sequential()
        self.model.add(Merge(embedding_models, mode='concat', concat_axis=1))
        self.model.add(LSTM(64, return_sequences=True, activation=Config.lstm_activation))
        if Config.lstm_dropout > 1e-10:
            self.model.add(Dropout(Config.lstm_dropout))
        self.model.add(LSTM(64, return_sequences=True, activation=Config.lstm_activation))
        if Config.lstm_dropout > 1e-10:
            self.model.add(Dropout(Config.lstm_dropout))
        self.model.add(LSTM(64, return_sequences=True, activation=Config.lstm_activation))
        if Config.lstm_dropout > 1e-10:
            self.model.add(Dropout(Config.lstm_dropout))
        self.model.add(LSTM(32, return_sequences=True, activation=Config.lstm_activation))
        self.model.add(LSTM(1, activation='softmax'))

    def __build_convolutional_model(self):
        """
        Builds a convolutional model, saves its configuration into file and compiles the model.
        :return:
        """
        print('\n\033[92mBuilding convolutional model...\033[0m')
        embedding_models = self.__get_embeddigs()
        self.model = Sequential()
        self.model.add(Merge(embedding_models, mode='concat', concat_axis=1))

        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation=Config.conv_activation, strides=1))
        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation=Config.conv_activation, strides=1))
        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation=Config.conv_activation, strides=1))
        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation=Config.conv_activation, strides=1))
        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation=Config.conv_activation, strides=1))

        self.model.add(Flatten())
        self.model.add(Dense(32, activation=Config.conv_activation))
        self.model.add(Dense(2, activation='softmax'))

    def __compile_model(self):
        """
        Compiles the model based on its type.
        :return:
        """
        if self.network_type == NetworkTypes.RECURRENT:
            self.model.compile(optimizer=Config.lstm_optimizer, loss=Config.lstm_loss, metrics=['accuracy'])
        else:
            self.model.compile(optimizer=Config.conv_optimizer, loss=Config.conv_loss, metrics=['accuracy'])

    def __get_embeddigs(self):
        """
        Creates a list of embedding layers for the neural networks. An embedding layer is corresponds to a word feature
        separately for every word in a sentence window.
        :return:
        """
        embedding_models = []
        max_feature_values = self.data_handler.get_tag_vocabulary_sizes()
        print(max_feature_values)

        # add an embedding layer for each feature of each word in a window
        for i in range(Config.window):
            for feature_value in max_feature_values:
                feature_model = Sequential()
                feature_model.add(Embedding(input_dim=feature_value,
                                            output_dim=Config.embedding_output_dim,
                                            input_length=1,
                                            trainable=True))
                feature_model.build()
                embedding_models.append(feature_model)
        return embedding_models

    def __save_model_configuration(self):
        """
        Saves the configuration of the neural network: the summary, patience of early stopping, the used optimizer,
        loss function, activation function and dropout. The file name will contain the type of network, the time of
        building the model and 'log'.
        """
        file_name = self.get_file_name()
        file_path = os.path.join(self.save_model_dir, file_name + '-log.txt')

        with open(file_path, mode='w') as save_file:
            try:
                # from Keras 2.0.6
                self.model.summary(print_fn=lambda x: save_file.write(x + '\n'))
            except Exception:
                # writing summary to file with older versions of Keras
                original_stdout = sys.stdout
                sys.stdout = save_file
                self.model.summary()
                sys.stdout = original_stdout

            save_file.write('\nEmbedding output dimension:\t' + str(Config.embedding_output_dim))
            save_file.write('\nEarly stopping patience:\t' + str(Config.patience))

            if self.network_type == NetworkTypes.RECURRENT:
                save_file.write('\nOptimizer:\t' + Config.lstm_optimizer)
                save_file.write('\nLoss:\t' + Config.lstm_loss)
                save_file.write('\nActivation:\t' + Config.lstm_activation)
                save_file.write('\nDropout:\t' + str(Config.lstm_dropout))
            else:
                save_file.write('\nOptimizer:\t' + Config.conv_optimizer)
                save_file.write('\nLoss:\t' + Config.conv_loss)
                save_file.write('\nActivation:\t' + Config.conv_activation)

    def __get_filename_time(self):
        """
        Returns the time of building the model. Using this method allows the files created at different times to include
        the same date and time, this way the files of the same network will stick together by name. In case the time of
        building the network isn't set, it sets it and returns the actual time.
        :return: The time of building the model.
        """
        if self.model_build_time is None:
            # save the actual time if it there is no saved build time
            self.model_build_time = datetime.now()
        return self.model_build_time
