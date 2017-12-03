import numpy as np
import sys

from math import log
from morph_disamb.config import Config


class MorphologicalDisambiguator:
    __slots__ = ('neural_network', 'data_handler', 'disamb_correct_word', 'disamb_word', 'disamb_correct_sentence',
                 'result_file', 'analysis_vect_to_string', 'evaluation', 'disamb_sentence', 'prev_correct_sentence')

    def __init__(self, deep_neural_network, disamb_file=None):
        self.neural_network = deep_neural_network
        self.data_handler = deep_neural_network.data_handler
        self.result_file = disamb_file
        self.analysis_vect_to_string = self.data_handler.data_loader.analysis_vect_to_string

        self.disamb_correct_word = 0
        self.disamb_word = 0
        self.disamb_correct_sentence = 0
        self.disamb_sentence = 0
        self.prev_correct_sentence = 0

        self.evaluation = False

    def disambiguate_sentences(self, evaluation=True):
        """
        Disambiguates the loaded sentences.
        :param evaluation: If the mode if evaluation, the statistics are counted and the expected analyses are shown.
        :return:
        """
        print('\n\033[92mDisambiguate sentences...\033[0m')
        self.evaluation = evaluation

        sentence_count = self.data_handler.get_sentences_to_disambiguate_count()
        print('Number of sentences:', sentence_count)
        if sentence_count < 1:
            return
        self.disamb_correct_word = 0
        self.disamb_word = 0
        self.disamb_sentence = 0
        self.disamb_correct_sentence = 0
        only_evaluation = Config.to_evaluate and not Config.to_disambiguate

        output = open(self.result_file, mode='w', encoding='utf-8') if self.result_file is not None else sys.stdout
        try:
            if self.result_file is not None:
                print(self.result_file)
            for sentence_idx in range(sentence_count):
                sentence_data = self.data_handler.get_all_combinations_for_sentence(sentence_idx, only_evaluation)
                if sentence_data is not None and len(sentence_data) > 0:
                    sentence_matrix = self.__process_sentence(sentence_data)
                    self.__run_viterbi(sentence_matrix, sentence_data[0], output, sentence_data[2], sentence_data[3])

                    if evaluation:
                        print("{0}\tword: {1} %\tsentence: {2} %".format(sentence_idx,
                                                                        100 * self.disamb_correct_word / self.disamb_word,
                                                                        100 * self.disamb_correct_sentence / self.disamb_sentence))

            print('Disambiguation ready!')
        finally:
            try:
                if evaluation:
                    output.write('\nWords:\n')
                    output.write('\tCorrect :\t' + str(self.disamb_correct_word) + '\n')
                    output.write('\tAll     :\t' + str(self.disamb_word) + '\n')
                    output.write('\t' + str(100 * self.disamb_correct_word / self.disamb_word) + ' %\n')

                    output.write('\nSentences:\n')
                    output.write('\tCorrect :\t' + str(self.disamb_correct_sentence) + '\n')
                    output.write('\tAll     :\t' + str(self.disamb_sentence) + '\n')
                    output.write('\t' + str(100 * self.disamb_correct_sentence / self.disamb_sentence) + ' %\n')
            finally:
                if self.result_file is not None:
                    output.close()

    def __process_sentence(self, sentence_data):
        """
        Feeds sentence into the network and parses the prediction values of the neural network into list of lists.
        The elements of the outer list correspond to the windows of the sentence, and the elements of the
        inner list are the probabilities of the combinations.
        :param sentence_data: Sentence input for the neural network.
        :return: The parsed and reformatted probabilities of combinations within the windows.
        """
        probabilities = self.neural_network.predict_data(list(np.concatenate(np.array(sentence_data[0]), axis=1).T))
        matrix = []
        combination_lengths = sentence_data[1]
        probability_index = 0
        for combination in combination_lengths:
            next_row = []
            for i in range(combination):
                next_row.append(probabilities[probability_index][-1])
                probability_index += 1
            matrix.append(next_row)
        return matrix

    def __run_viterbi(self, probabilities_matrix, sentence_data, result_file, words, analyses):
        """
        Creates Viterbi matrices, determines the most probable sequence and writes it to the requested output.
        :param probabilities_matrix: The probabilities of combinations in the windows.
        :param sentence_data: The windows and combinations in the sentence.
        :param result_file: Where the result has to be written.
        :param words: The words of the sentence as strings.
        :param analyses: The correct analyses of the words (used in case of evaluation).
        :return:
        """
        self.prev_correct_sentence = self.disamb_correct_sentence
        viterbis_in_sentence = self.__get_viterbis_of_sentence(probabilities_matrix, sentence_data)
        viterbi_result = self.__get_viterbi_result(viterbis_in_sentence)
        self.__save_analysis_from_viterbi_result(viterbi_result, result_file, words, analyses)

    def __get_viterbis_of_sentence(self, probabilities_matrix, sentence_data):
        """
        Creates Viterbi matrices with Viterbi values and backpointers from probabilities of combinations.
        :param probabilities_matrix: The probabilities of combinations in the windows.
        :param sentence_data: The windows and combinations in the sentence.
        :return: Viterbi matrices of the sentence.
        """
        matrix_idx = 0
        viterbis_in_sentence = []
        analyses_idx = 0

        # iterate through windows in sentence
        for row in probabilities_matrix:
            viterbi = {}
            viterbi_window = {}
            correct = None

            # iterate through every analysis combination in the window
            for col in row:
                combination = tuple(tuple(element[analyses_idx]) for element in sentence_data)
                if correct is None:
                    correct = combination[-1]
                analyses_idx += 1
                viterbi_window.setdefault(combination[1:], [])
                # find matching line from previous viterbi matrix if it exists
                matched = self.__match_last_viterbi(viterbis_in_sentence[matrix_idx - 1] if matrix_idx > 0 else None,
                                                    combination)
                matched['viterbi_value'] += log(col + 1e-10)
                matched['last_correct'] = (correct == combination[-1])
                viterbi_window[combination[1:]].append(matched)

            # find max backpointer viterbi value among same ending analyses
            for combination in viterbi_window:
                max_viterbi_combination = viterbi_window[combination][0]
                for backpointer in viterbi_window[combination]:
                    if backpointer['viterbi_value'] > max_viterbi_combination['viterbi_value']:
                        max_viterbi_combination = backpointer

                # save the combination belonging to max viterbi value backpointer
                viterbi[combination] = max_viterbi_combination

            viterbis_in_sentence.append(viterbi)
            matrix_idx += 1

        return viterbis_in_sentence

    def __get_viterbi_result(self, viterbis_in_sentence):
        """
        Determines the optimal sequence of windows of analysis combinations based on Viterbi matrices.
        It runs the back stepping phase of the algorithm using the backpointers.
        :param viterbis_in_sentence: The Viterbi matrices which include Viterbi values and backpointers for each combination
        :return: The optimal window combination sequence
        """
        viterbi_result = []

        # find best viterbi value in last matrix
        max_viterbi_combination = ()
        for combination in viterbis_in_sentence[-1]:
            if not max_viterbi_combination or viterbis_in_sentence[-1][combination]['viterbi_value'] > \
                    viterbis_in_sentence[-1][max_viterbi_combination]['viterbi_value']:
                max_viterbi_combination = combination

        # check word correctness in last combination
        disamb_word = 1
        disamb_correct_word = 1 if viterbis_in_sentence[-1][max_viterbi_combination]['last_correct'] else 0

        next_combination = viterbis_in_sentence[-1][max_viterbi_combination]['backpointer']

        # when the sentence consists of more than 1 word, also save the next combination
        if len(viterbis_in_sentence) > 1:
            viterbi_result.append(next_combination)
        viterbi_result.append(max_viterbi_combination)

        # stepping back on backpointers
        for viterbi in viterbis_in_sentence[-2:0:-1]:
            disamb_word += 1
            if viterbi[next_combination]['last_correct']:
                disamb_correct_word += 1

            next_combination = viterbi[next_combination]['backpointer']
            viterbi_result.insert(0, next_combination)

        self.disamb_correct_word += disamb_correct_word
        self.disamb_word += disamb_word
        self.disamb_sentence += 1
        self.disamb_correct_sentence += 1 if disamb_word == disamb_correct_word else 0

        # print(disamb_word, disamb_correct_word, self.disamb_sentence, self.disamb_correct_sentence)

        return viterbi_result

    def __save_analysis_from_viterbi_result(self, viterbi_result, result_file, words, analyses):
        """
        Decodes resulting analysis vectors to analysis strings and writes them into file. Also writes the expected
        analysis strings if an evaluation is executed, therefore the right analyses are
        :param viterbi_result: Windows of analysis combinations of the given sentence with the highest probability
        :param result_file: Where the decoded disambiguated analyses have to be written.
        :param words: The word strings of the sentence.
        :param analyses: The correct analysis strings in the sentence.
        :return:
        """
        # save last analysis for combinations
        for i in range(len(viterbi_result)):
            word_analyses = self.analysis_vect_to_string[words[i+4]]
            viterbi_analysis = viterbi_result[i][-1]

            got_analyses = '\t'.join(word_analyses[viterbi_analysis])
            if self.evaluation:
                result_file.write(words[i+4] + '\n')
                result_file.write('\texpected :\t' + '\t'.join(analyses[i]) + '\n')
                result_file.write('\tgot      :\t' + got_analyses)

            else:
                result_file.write(got_analyses)
            result_file.write('\n')

        result_file.write('\n')

    def __match_last_viterbi(self, last_viterbi, actual_combination):
        """
        Gets the matching element from the last Viterbi matrix. The last four analyses in the matching combination must
        be equal to the first four analyses of the current one. If the last matrix is empty, the first four analyses
        are returned with no backpointer.
        :param last_viterbi: The previous Viterbi matrix.
        :param actual_combination: The current analysis combination.
        :return: a dictionary of the Viterbi value and the backpointer
        """
        viterbi_value = 0
        if last_viterbi is not None:
            viterbi_value = last_viterbi[actual_combination[:-1]]['viterbi_value']

        return {'viterbi_value': viterbi_value,
                'backpointer': actual_combination[:-1]}
