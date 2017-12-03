import itertools
import numpy as np

from morph_disamb.config import Config


class NeuralNetworkDataHandler:
    __slots__ = ('training_data', 'evaluation_data', 'training_labels', 'evaluation_labels', 'data_loader')

    def __init__(self, data_loader):
        self.training_data = []
        self.evaluation_data = []
        self.training_labels = []
        self.evaluation_labels = []

        self.data_loader = data_loader

    def create_network_input_data(self):
        """
        Creates training and evaluation inputs from sentences.
        :return:
        """
        print('\n\033[92mCreating neural network input data...\033[0m')
        self.__init_network_input_data()
        positive_input_count = 0
        sentence_counter = 0
        # the number of sentences which ha to be used for training
        training_sentence_count = self.data_loader.get_training_sentence_count()
        print("Count of training sentences     :\033[96m", training_sentence_count, '\033[0m')
        print("Count of evaluation sentences   :\033[96m", len(self.data_loader.sentences) - training_sentence_count, '\033[0m')

        for sentence in self.data_loader.sentences:
            sentence_counter += 1
            for word_idx in range(Config.window-1, len(sentence)):
                # getting the good analyses of the last window-2 words
                next_input = []
                for earlier_word_idx in range(word_idx - Config.window + 1, word_idx - 1):
                    next_input.append(sentence[earlier_word_idx]['analyses'][0])

                # add every combination of the last two words in the window to the next input
                for i in range(len(sentence[word_idx-1]['analyses'])):
                    for j in range(len(sentence[word_idx]['analyses'])):
                        next_input.append(sentence[word_idx - 1]['analyses'][i])
                        next_input.append(sentence[word_idx]['analyses'][j])

                        # the first analyses of both words are the correct ones
                        is_positive = (i == 0 and j == 0)
                        self.__add_next_input(next_input, is_positive, sentence_counter, training_sentence_count)

                        # counting positive examples
                        if is_positive:
                            positive_input_count += 1

                        # remove the last two analyses, so that only window-2 analyses stay in the list
                        next_input.pop()
                        next_input.pop()

        print("Count of positive examples      :\033[96m", positive_input_count, '\033[0m')
        print("Count of training examples      :\033[96m", len(self.training_labels), '\033[0m')
        print("Count of evaluation examples    :\033[96m", len(self.evaluation_labels), '\033[0m')

    def get_all_combinations_for_sentence(self, sentence_index, only_evaluation):
        """
        Collect every neural network input of the given sentence.
        :param sentence_index: The index of the sentence whose windows are requested.
        :param only_evaluation: True when all of the sentences should be used for evaluation, not only the last small part
        :return: The collected combinations of the sentence, the number of combinations within a window, the word
        strings in the sentence and the good analysis strings
        """
        # if the data is not split into training and evaluation parts, all the input sentences are used for evaluation,
        # otherwise only the last part of it
        if only_evaluation:
            sentence_index += round(len(self.data_loader.sentences) * Config.training_percent)

        if sentence_index < 0 or sentence_index >= len(self.data_loader.sentences):
            return []

        sentence = self.data_loader.sentences[sentence_index]
        sentence_combination_lengths = []   # the number of combinations in the windows of the sentence
        sentence_data = []
        good_analyses = []
        for i in range(Config.window):
            sentence_data.append([])

        # iterating through windows in the sentence by index
        for word_idx in range(Config.window-1, len(sentence)):
            combination_count = 1
            window_analyses = []

            # iterating through the words of the window
            for i in range(word_idx - Config.window + 1, word_idx + 1):
                # counting the possible combinations of the actual window
                combination_count *= len(sentence[i]['analyses'])
                # collecting the analyses of the words in the window for creating combinations
                window_analyses.append(sentence[i]['analyses'])

            # collect correct analysis strings for words
            good_analysis = tuple(window_analyses[-1][0])
            word = sentence[word_idx]['string']
            good_analyses.append(self.data_loader.analysis_vect_to_string[word][good_analysis])

            sentence_combination_lengths.append(combination_count)
            # every combination of the analysis in the window
            combinations_of_window = list(itertools.product(*window_analyses))

            for combination in combinations_of_window:
                for i in range(Config.window):
                    sentence_data[i].append(combination[i])

        return [sentence_data,
                sentence_combination_lengths,
                [word['string'] for word in sentence],
                good_analyses]

    def get_training_array(self):
        """
        :return: The training data as list of numpy arrays.
        """
        return list(np.concatenate(np.array(self.training_data), axis=1).T)

    def get_evaluation_array(self):
        """
        :return: The evaluation data as list of numpy arrays.
        """
        return list(np.concatenate(np.array(self.evaluation_data), axis=1).T)

    def get_tag_vocabulary_sizes(self):
        """
        :return:  The biggest occurring ID numbers in the tag categories.
        """
        return self.data_loader.count_max_analyses_values()

    def get_sentences_to_disambiguate_count(self):
        """
        :return: The number of sentences which can be used for disambiguation. Equals to the number of sentences
        if the data is not split into training and evaluation data, otherwise to the number of evaluation sentences.
        """
        if Config.to_evaluate and not Config.to_disambiguate:
            return len(self.data_loader.sentences) - round(len(self.data_loader.sentences) * Config.training_percent)
        return len(self.data_loader.sentences)

    def __add_next_input(self, next_input, is_positive, sentence_count, max_training_count):
        """
        Adds the next neural network input to the proper arrays.
        If the sentence index is not greater than the total number of sentences which has to be used for training,
        then the new input is added to the training data, otherwise to the evaluation data.
        :param next_input: the next input
        :param is_positive: is it a positive example
        :param sentence_count: the index of the actually processed sentence
        :param max_training_count: the number of training sentences
        :return:
        """
        if sentence_count <= max_training_count:
            for k in range(Config.window):
                self.training_data[k].append(next_input[k])
            self.training_labels.append([1.0 if is_positive else 0.0])
        else:
            for k in range(Config.window):
                self.evaluation_data[k].append(next_input[k])
            self.evaluation_labels.append([1.0 if is_positive else 0.0])

    def __init_network_input_data(self):
        """
        Initializes neural network input data. Clears the training and evaluation data lists and the labels,
        then adds window number of empty arrays to the input data arrays.
        :return:
        """
        self.training_data = []
        self.evaluation_data = []

        self.training_labels = []
        self.evaluation_labels = []

        # adding window number of lists to training and evaluating data (because of neural network merging)
        for i in range(Config.window):
            self.training_data.append([])
            self.evaluation_data.append([])
