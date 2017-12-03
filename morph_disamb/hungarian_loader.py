import os
import re
import subprocess
import tempfile
import platform

from morph_disamb.common_funcs import get_file_paths
from morph_disamb.config import Config

from random import shuffle, seed
from pprint import pprint



class HungarianDataLoader:
    __slots__ = ('features', 'categories', 'category_indexes', 'unprocessed_features', 'sentences',
                 'analyses_vect_of_words', 'preverbs', 'new_root_added', '__next_root_value',
                 '__sentence_separator_analysis', 'not_found_features', 'analysis_vect_to_string',
                 'analyses_of_words', '__unknown_analysis', 'input_data')

    def __init__(self):
        self.sentences = []
        self.input_data = []

        self.features = {}                              # tag - its category
        self.categories = {'root': {Config.unknown_root: 1,
                                    Config.sentence_separator: 2,
                                    Config.number: 3}}  # feature category - included tags - tag IDs
        self.category_indexes = {'root': 0}             # feature category - ID
        self.unprocessed_features = []
        self.not_found_features = []
        self.analyses_vect_of_words = {}
        self.preverbs = []
        self.analysis_vect_to_string = {}

        self.new_root_added = False
        self.__next_root_value = 4  # 0 - root does not exist, 1 - UNK, 2 - <S>, 3 - NUM
        self.__sentence_separator_analysis = []
        self.__unknown_analysis = []

        self.__load_features()

    def load_input_from_file(self, corpus_dir, corpus_file=None):
        """
        Loads corpus from the given file(s), then collects analyses for the words.
        :param corpus_dir: The directory where the corpus can be found.
        :param corpus_file: The file from which the corpus should be loaded. If it is not given, all the files
        would be loaded from the given directory.
        :return:
        """
        print('\n\033[92mLoading Hungarian corpus...\033[0m')
        print('Corpus directory: \033[95m', corpus_dir, '\033[0m')
        files_to_load = get_file_paths(corpus_dir, corpus_file)     # file paths
        print('Loading\033[95m', len(files_to_load), '\033[0mfile(s)')

        self.input_data = []

        # each file gets loaded
        for file_path in files_to_load:
            with open(file_path, mode='r', encoding='utf-8') as f:
                print(file_path)
                for line in f:
                    split_line = line.strip().split('\t')
                    # if the split line is empty and the first element is empty, it was a sentence separator
                    if len(split_line) > 0 and len(split_line[0]) > 0:
                        word = split_line[0]
                        # saving word string and good analysis if it can be found
                        if len(split_line) > 1:
                            good_analysis = split_line[-1]
                            self.input_data.append([word, good_analysis])
                        else:
                            self.input_data.append([word])
                    else:
                        self.input_data.append([])
                if len(self.input_data[-1]) > 0:
                    self.input_data.append([])

        print('\n\033[92mCreating sentence data...\033[0m')
        self.__create_sentences(shuffle_training=True)

    def tokenize_input_text(self, text):
        """
        Tokenizes the input text with quntoken and collects word analyses for the sentence.
        :param text: The text which should be tokeized.
        :return:
        """
        self.input_data = []

        with tempfile.NamedTemporaryFile() as temp:
            temp.write(text.encode())
            temp.flush()

            quntoken = subprocess.Popen(
                'quntoken -d ' + temp.name,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            output, err = quntoken.communicate()

            err = err.decode('utf-8')
            if len(err) > 0:
                print('\033[91m' + err + '\033[0m')
            else:
                output = output.decode('utf-8').strip()
                output = re.sub('<\/s>', '\n', output)      # separating sentences into lines
                output = re.sub('<\/[wc]>', '\n', output)   # separating words and punctuations into lines
                output = re.sub('<.*>', '', output)         # deleting whitespace and other remaining tags

                # collecting words and sentence separator empty lines from the tokenized text
                self.input_data = [([word.strip()] if len(word.strip()) > 0 else []) for word in output.split('\n')]

        self.__create_sentences()

    def save_root_vocabulary_to_file(self):
        """
        Saves the root vocabulary to file in the following format: <root><tabulator><root value in analysis vector>
        :param saving_dir: The directory for saving.
        :param saving_file: The file name for saving.
        :return:
        """
        print('\n\033[92mSaving root vocabulary...\033[0m')
        save_file_path = Config.root_file
        print('File path: \033[95m', save_file_path, '\033[0m')

        with open(save_file_path, mode='w', encoding='utf-8') as save:
            for root in sorted(self.categories['root']):
                save.write('{0}\t{1}\n'.format(root, self.categories['root'][root]))

    def load_root_vocabulary_from_file(self, clear_before_load=True):
        """
        Loads the root vocabulary from file. In the file every line has to contain the data of one root in the
        following format: <root><tabulator><root value in analysis vector>
        :param load_dir: The directory where the root vocabulary can be found.
        :param load_file: The file name of the root vocabulary file.
        :param clear_before_load: Whether or not the existing root vocabulary has to be cleared.
        :return:
        """
        print('\n\033[92mLoading root vocabulary...\033[0m')
        if clear_before_load:
            self.categories['root'] = {Config.unknown_root: 1,
                                       Config.sentence_separator: 2,
                                       Config.number: 3}

        load_file_path = Config.root_file
        print('File path:\033[95m', load_file_path, '\033[0m')

        if os.path.isfile(load_file_path):
            with open(load_file_path, mode='r', encoding='utf-8') as load:
                for line in load:
                    split_line = line.split('\t')
                    if len(split_line) == 2:
                        self.categories['root'][split_line[0]] = int(split_line[1])
                    else:
                        print('Bad file format: ', load_file_path)
                        return

        # in case of expanding the next root value will be needed, which is counted after loading the vocabulary
        self.__count_next_root_value()

    def load_possible_word_analyses(self, file_path, to_clean=False):
        """
        Loads the possible word analyses, and stores them as vectors. Cleans the stored data before if it's necessary.
        :param file_path: the path for the analyses file.
        :param to_clean: Whether the data store has to be cleared beforehand or not.
        :return:
        """
        print('\n\033[92mLoading word analyses...\033[0m')
        print('File path: \033[95m', file_path, '\033[0m')
        if to_clean:
            self.analyses_vect_of_words = {}

        if os.path.isfile(file_path):
            with open(file_path, mode='r', encoding='utf-8') as analyses_file:
                for line in analyses_file:
                    split_line = line.split()
                    if len(split_line) > 1:
                        word = split_line[0]
                        analysis_string = split_line[1]
                        self.__save_word_analysis(word, analysis_string)

        print("Word count: ", len(self.analyses_vect_of_words.keys()))

    def count_max_analyses_values(self):
        """
        Collects the maximum occurring numbers in the tag categories separately.
        :return: The maximum values ordered by the category indexes in the analysis vectors.
        """
        max_analysis_values = []
        # keys of the categories ordered by their index in the analysis vectors
        keys = sorted(self.category_indexes, key=self.category_indexes.get)
        for k in keys:
            # maximum value in a category
            max_feature_value = max(self.categories[k], key=self.categories[k].get)
            max_analysis_values.append(self.categories[k][max_feature_value] + 1)
        return max_analysis_values

    def print_statistics(self):
        """
        Prints some statistics about the data.
        :return:
        """
        sentence_number = len(self.sentences)
        word_number = 0
        analyses_number = 0
        LOT_ANALYSIS = 10
        LONG_SENTENCE = 200

        sentence_length_stat = {}
        analysis_count_stat = {}
        lot_of_analysis = {}
        long_sentences = []

        for sentence in self.sentences:
            sentence_length = len(sentence) - 4
            word_number += sentence_length

            # counting sentence length frequency
            if sentence_length not in sentence_length_stat:
                sentence_length_stat[sentence_length] = 1
            else:
                sentence_length_stat[sentence_length] += 1

            # collecting long sentences
            if sentence_length > LONG_SENTENCE:
                long_sentences.append(sentence)

            for word in sentence:
                if word['string'] == Config.sentence_separator:
                    continue

                # counting the number of analyses of a word
                analyses_count2 = len(word['analyses'])
                if word['string'] in self.analysis_vect_to_string:
                    analyses_count = len(self.analysis_vect_to_string[word['string']])
                else:
                    analyses_count = 1
                if analyses_count != analyses_count2:
                    print(self.analysis_vect_to_string[word['string']])

                analyses_number += analyses_count
                # collecting words with a lot of analyses
                if analyses_count > LOT_ANALYSIS:
                    lot_of_analysis[word['string']] = analyses_count

                # counting word analysis number frequency
                if analyses_count not in analysis_count_stat:
                    analysis_count_stat[analyses_count] = 1
                else:
                    analysis_count_stat[analyses_count] += 1

        print("Number of sentences:", sentence_number)
        print("Number of words:", word_number)
        print("Average number of words per sentence:", word_number / sentence_number)
        print("Average number of analyses per word with frequence:", analyses_number / word_number)
        print("Sentence length stat:")
        pprint(sentence_length_stat)
        print("Analyses count stat:")
        pprint(analysis_count_stat)
        print("Lof of analyses:")
        pprint(lot_of_analysis)

    def get_training_sentence_count(self):
        """
        Counting the number of training sentences which equals to the number of sentences * training percent.
        :return: The number of training sentences.
        """
        training_sentence_count = round(len(self.sentences) * Config.training_percent)
        if training_sentence_count == len(self.sentences):
            training_sentence_count -= 1
        return training_sentence_count

    def __load_features(self):
        """
        Loads the feature categories and their possible values from the given file. Assigns vector index value to each
        category which tells that in a word analysis vector which position has the category. Also assigns unique number
        for the possible tags of the categories.
        :param feature_dir: The directory of the features file.
        :param feature_file: The name of the features file.
        :return:
        """
        print('\n\033[92mLoading features...\033[0m')
        print('File path: \033[95m', Config.features_file, '\033[0m')

        # opening file which contains the used features
        # category indexes start from 1, because 0 is assigned to the root
        category_index = 1
        with open(Config.features_file, 'r', encoding='utf-8') as feature_f:
            for line in feature_f:
                split_line = str.split(line)
                if len(split_line) > 0:
                    # the first element in the line is the feature category
                    category = split_line[0].lower()
                    self.categories[category] = {}

                    # giving category index to the new category
                    self.category_indexes[category] = category_index
                    category_index += 1

                    # for every feature of the category, store the relationship
                    feature_index = 1
                    for i in range(1, len(split_line)):
                        feature = split_line[i].lower()
                        if feature in self.features:
                            print('\033[91m', feature, '\033[0m already in category', self.features[feature],
                                  'new category would be', category)
                        else:
                            self.features[feature] = category
                        self.categories[category][feature] = feature_index
                        feature_index += 1

        # saving the analysis vector of the sentence separator and unknown analysis
        self.__sentence_separator_analysis = self.__get_analysis_vector(Config.sentence_separator)
        self.__unknown_analysis = self.__get_analysis_vector(Config.unknown_root)

        self.analysis_vect_to_string[Config.sentence_separator] = \
            {tuple(self.__sentence_separator_analysis): [Config.sentence_separator]}
        self.analysis_vect_to_string[Config.unknown_root] = \
            {tuple(self.__unknown_analysis): [Config.unknown_root]}

    def __create_sentences(self, shuffle_training=False):
        """
        Creates sentence data from input data. Separates the input into sentences and collects the analyses of the words.
        :param shuffle_training: Whether or not to shuffle the training part of sentences.
        :return:
        """
        self.sentences = []
        next_sentence = self.__start_new_sentence([])

        for word_data in self.input_data:
            if len(word_data) > 0:
                word = word_data[0]
                good_analysis = None
                if len(word_data) > 1:
                    good_analysis = word_data[-1]
                analyses = self.__get_word_analyses(word, good_analysis)
                next_sentence.append({
                    'string': word,
                    'analyses': analyses
                })
            else:
                # empty line means the start of a new sentence
                next_sentence = self.__start_new_sentence(next_sentence)
        # saving the last sentence data
        self.__start_new_sentence(next_sentence)

        # shuffling the training sentences
        if shuffle_training:
            seed(2017)
            training_sentence_count = self.get_training_sentence_count()
            training_sentences = self.sentences[:training_sentence_count]
            shuffle(training_sentences)
            self.sentences = training_sentences + self.sentences[training_sentence_count:]

    def __get_word_analyses(self, word, good_analysis=None):
        """
        Collects every possible analyses for a given word and returns them as a list. The first will be the good
        analysis for the word which is got from the parameter analysis. Every analysis is only added once.
        :param word: The analyses of these word are collected.
        :param good_analysis: The analysis string of the good analysis of the word.
        :return: The list of word analyses.
        """
        # analyze word with HFST if no analysis is available
        if word not in self.analyses_vect_of_words:
            self.__analyze_word(word)

        # if the good analysis was given, search the best matching analysis and put it at the beginning of the returned list
        if good_analysis is not None:
            good_analysis_vector = self.__get_analysis_vector(good_analysis)

            # find the best matching analysis in HFST analyses
            if word in self.analyses_vect_of_words:
                stored_analyses = self.analyses_vect_of_words[word]
                best_score = 0
                nearest_vector = None

                # for each analyses of the word, count how many fields are equal and keep the best
                for analysis_vector in stored_analyses:
                    same_field_count = 0
                    for i in range(len(good_analysis_vector)):
                        if good_analysis_vector[i] == analysis_vector[i]:
                            same_field_count += 1
                    if same_field_count > best_score or nearest_vector is None:
                        best_score = same_field_count
                        nearest_vector = analysis_vector

                # putting the best match to the beginning of the result list, and adding all the others
                nearest_idx = stored_analyses.index(nearest_vector)
                analyses = [nearest_vector] + stored_analyses[:nearest_idx] + stored_analyses[nearest_idx+1:]
            else:
                # if the word is not found even after analyzing it with HFST, the good analysis will be saved as it is
                analyses = [good_analysis_vector]
            return analyses
        else:
            if word in self.analyses_vect_of_words:
                # if no good analysis was given, return all the found analyses
                return self.analyses_vect_of_words[word]
            elif word.lower() in self.analyses_vect_of_words:
                # try to find the lowercase version of the word
                self.analysis_vect_to_string[word] = self.analysis_vect_to_string[word.lower()]
                return self.analyses_vect_of_words[word.lower()]
            else:
                # if nothing works, the word has an unkown analysis
                return [self.__unknown_analysis]

    def __analyze_word(self, word):
        """
        Analyzes a word using HFST and saves it to a dictionary.
        :param word: The word to be analyzed.
        :return:
        """
        if platform.system() == 'Linux':
            lookup = 'hfst-lookup'
        elif platform.system() == 'Windows':
            lookup = 'hfst-lookup.exe'
        else:
            # the analysis is the word itself -> the root will be the word + no other features
            self.__save_word_analysis(word, word)
            return

        hfst = subprocess.Popen(
            lookup + ' --pipe-mode=input --cascade=composition -s "' + Config.transducer + '"',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )

        output, err = hfst.communicate(word.encode())

        err = err.decode('utf-8')
        if len(err) > 0:
            print('\033[91m' + err + '\033[0m')
            # the analysis is the word itself -> the root will be the word + no other features
            self.__save_word_analysis(word, word)
        else:
            output = output.decode('utf-8')

            for line in output.splitlines():
                split_line = line.split('\t')
                # the first element of the split line is the word, the second is the analysis
                if len(split_line) >= 2:
                    analysis_string = split_line[1]
                    self.__save_word_analysis(word, analysis_string)

    def __save_word_analysis(self, word, analysis_string):
        """
        Saves the analysis string and vector of a word.
        :param word: The word whose analysis is given.
        :param analysis_string: The analysis string of the word.
        :return:
        """
        analysis_vector = self.__get_analysis_vector(analysis_string)

        # adding analysis vector to the analysis vector collection of the word
        if word in self.analyses_vect_of_words:
            if analysis_vector not in self.analyses_vect_of_words[word]:
                self.analyses_vect_of_words[word].append(analysis_vector)
        else:
            self.analyses_vect_of_words[word] = [analysis_vector]

        # saving analysis vector - analysis string mapping of the word
        analysis_vect_tuple = tuple(analysis_vector)
        if word not in self.analysis_vect_to_string:
            self.analysis_vect_to_string[word] = {}

        if analysis_vect_tuple not in self.analysis_vect_to_string[word]:
            self.analysis_vect_to_string[word][analysis_vect_tuple] = []

        if analysis_string not in self.analysis_vect_to_string[word][analysis_vect_tuple]:
            self.analysis_vect_to_string[word][analysis_vect_tuple].append(analysis_string)

    def __start_new_sentence(self, next_sentence):
        """
        Saves the data of a sentence if it is not empty, then creates a new sentence beginning by adding window-1
        sentence separator analyses to a new array.
        :param next_sentence: The sentence which has to be saved if it's not empty.
        :return: The list for a new sentence: window-1 sentence separators added.
        """
        if len(next_sentence) > Config.window - 1:
            self.sentences.append(next_sentence)
        next_sentence = []
        for i in range(Config.window - 1):
            next_sentence.append({'string': Config.sentence_separator,
                                  'analyses': [self.__sentence_separator_analysis]})
        return next_sentence

    def __count_next_root_value(self):
        """
        Counts the next root value in case of expanding the root vocabulary. It will be one more than the maximum value
        found in the root vocabulary.
        :return:
        """
        max_root_value = max(list(self.categories['root'].values()))
        self.__next_root_value = max_root_value + 1
        print('Next root value:', self.__next_root_value)

    def __get_analysis_vector(self, analysis_string):
        """
        Parses the given word analysis string using the category and feature information.
        :param analysis_string: The word analysis string.
        :return: The analysis vector of the word.
        """
        analysis = [0 for category in self.categories]
        root, preverb = self.__get_root_and_preverb(analysis_string)

        # saving the root value for the analysis
        analysis[self.category_indexes['root']] = self.__get_root_value(root)

        # getting every feature which is between [ and ] signs and preprocessing them
        features = re.findall(r'\[.*?\]', analysis_string)
        features = self.__preprocess_features(features)

        # adding preverb feature separately because it is not between [ and ] signs
        if preverb is not None:
            features.append(preverb)
            if '[/prev]' in features:
                features.remove('[/prev]')

        # adding features to the analysis vector to the right position with the right value
        for feature in features:
            if feature in self.features:
                category_of_feature = self.features[feature]
                category_index = self.category_indexes[category_of_feature]     # the position of the feature value
                feature_value_in_category = self.categories[category_of_feature][feature]

                analysis[category_index] = feature_value_in_category
            elif feature not in self.not_found_features:
                self.not_found_features.append(feature)

        return analysis

    def __get_preverb(self, analysis_string):
        """
        Parses the preverb from the analysis string and returns it if it exists
        :param analysis_string: A word analysis string.
        :return: The preverb if it exists, None otherwise.
        """
        match = re.search(r'\w*\[/Prev\]\w*', analysis_string)
        if match is not None:
            parts = re.split(r'\[/Prev\]', match.group(0))
            return parts[0]

    def __get_root_value(self, root):
        """
        Determines the value for a root. If the root is not in the vocabulary which is expandable, it adds the new root,
        otherwise if the vocabulary is not expendable, the root is considered unknown.
        :param analysis_string: A word analysis string.
        :return: The value of the root in the analysis vector.
        """
        # getting root value
        if root in self.categories['root']:
            root_value = self.categories['root'][root]
        else:
            if not Config.static_roots:
                # adding new root if it is not prohibited
                self.new_root_added = True
                root_value = self.__next_root_value
                self.categories['root'][root] = root_value
                self.__next_root_value += 1
            else:
                # if the root is not in the vocabulary and it can't be added to is, it is considered unknown
                root_value = self.categories['root'][Config.unknown_root]

        return root_value

    def __get_root(self, analysis_string):
        """
        Parses the root string from a word analysis string.
        :param analysis_string: A word analysis string.
        :return: The root string.
        """
        # if the word is a verb with preverb, the root will be the part after the [/Prev] tag
        # <preverb>[/Prev]<root>[<other tag>]
        match = re.search(r'\[/Prev\]\w*', analysis_string)    # if the preverb shouldn't be part of the root
        # match = re.match(r'\w*\[/Prev\]\w*', analysis_string)   # if the preverb is part of the root
        if match is not None:
            root = match.group(0).replace('[/Prev]', '')
            return self.__replace_uknown_root(root)

        # if the word is a superlative verb, the root will be the part after the [/Supl] tag
        # leg[/Supl]<root>[<other tag>]
        match = re.search(r'\[/Supl\]\w*', analysis_string)
        if match is not None:
            root = match.group(0).replace('[/Supl]', '')
            return self.__replace_uknown_root(root)

        # if the word couldn't be analyzed, the root will be the part before the +?
        # <root>+?
        match = re.match(r'.*\+\?$', analysis_string)
        if match is not None:
            return analysis_string.replace('+?', '')

        # if none of the earlier is true, the root will be the string before the first tag, or if there aren't any tags,
        # it will be the whole analysis string
        # <root>[<some tag>] or <root>
        match = re.match(r'^[^\[]*', analysis_string)
        if match is not None:
            return self.__replace_uknown_root(match.group(0))
        else:
            return analysis_string

    def __replace_uknown_root(self, possible_root):
        """
        Replaces empty root with unknown token.
        :param possible_root: The root string which has to be checked.
        :return: The original root string if it is not empty, unknown token otherwise.
        """
        if possible_root is None or len(possible_root) == 0:
            return Config.unknown_root
        return possible_root

    def __get_root_and_preverb(self, analysis_string):
        """
        Parses root and preverb from an analysis string.
        :param analysis_string: The analysis which has to be parsed.
        :return: The root and preverb string.
        """
        # splitting by POS tags, except the [/Prev] and [/Supl] tags
        split_analysis = re.split(r'\[\/(?!(Prev|Supl))[^\]]*\]', analysis_string)
        split_analysis = list(filter(None, split_analysis))
        root_extraction_parts = split_analysis[:-1]  # the part after the last main POS won't contain root
        root = ''
        preverb = None

        if len(root_extraction_parts) <= 1:
            # not compound word - get root ans preverb separately
            root = self.__get_root(analysis_string)
            preverb = self.__get_preverb(analysis_string)
        else:
            # compound word - deleting remaining tags from root parts and concatenating them
            for root_part in root_extraction_parts:
                root += "".join(re.split(r'\[.*?\]', str(root_part)))

        # try to convert the root into number - first deleting some special characters which can be part of numbers
        try:
            float(''.join(re.split(r'[%,\.\-\/:_\s]', root)))
            root = Config.number    # successful conversion into a number implies a common number token as root
        except:
            pass

        return root, preverb

    def __preprocess_features(self, features):
        """
        Preprocesses feature list, executes some conversions for better categorization.
        :param features: The feature list.
        :return: The processed feature list.
        """
        processed_features = []
        concatenated = ''

        for feature in features:
            feature = feature.lower()
            feature = feature.replace('%', '')

            replace_feature_to = {'[v]': '[/v]',
                                  '[n]': '[/n]',
                                  '[num]': '[/num]',
                                  '[adj|nat]': '[/adj|nat]',
                                  '[inl]': '[loc]'}

            if feature in replace_feature_to:
                feature = replace_feature_to[feature]
            elif feature in ['[1]', '[2]', '[3]']:
                # part of a separated person tag - beginning part
                concatenated = feature[:-1]
                continue
            elif feature == '[s]':
                # part of a separated person tag - center part
                concatenated += feature[1:-1]
                continue
            elif feature in ['[sg]', '[g]', '[pl]']:
                # part of a separated person tag - end part
                if len(concatenated) > 0:
                    concatenated += feature[1:]
                    processed_features.append(concatenated)
                    concatenated = ""
                    continue

            if feature in self.features:
                processed_features.append(feature)
            else:
                # checking some cases of the functions
                check_functions = [self.__check_possessive,
                                   self.__check_formative,
                                   self.__check_main_minor,
                                   self.__check_dot_separation]

                matched = False
                for check_function in check_functions:
                    if not matched:
                        matched = check_function(feature, processed_features)

                if not matched:
                    self.unprocessed_features.append(feature)

        return processed_features

    def __check_possessive(self, feature, processed_features):
        """
        Separates possessive agreement into two parts - doesn't have to be separated at all dots
        Check format: [<text>.poss.<text>] second part - from the first dot
        :param feature: The feature which has to be processed.
        :param processed_features: The new feature set.
        :return: Whether the tag was matched by the pattern or not.
        """
        match = re.search(r'\.poss\.\w+\]', feature)
        if match is not None:
            possessive = '[' + match.group(0)[1:]
            plurality = feature[0:feature.index(match.group(0))] + ']'
            processed_features.append(possessive)
            processed_features.append(plurality)
            return True

    def __check_formative(self, feature, processed_features):
        """
        Separates formatives from POS tags
        Check format: [_<text>/<text>] pattern
        :param feature: The feature which has to be processed.
        :param processed_features: The new feature set.
        :return: Whether the tag was matched by the pattern or not.
        """
        match = re.match(r'\[_.*/.*\]', feature)
        if match is not None:
            parts = re.split('/', feature)
            processed_features.append(parts[0] + ']')
            processed_features.append('[/' + re.search('\w+', parts[1]).group(0) + ']')
            return True

    def __check_dot_separation(self, feature, processed_features):
        """
        Separates tag at all dots.
        Check format: [<text>.<text>] or [<text>.<text>.<text>] pattern
        :param feature: The feature which has to be processed.
        :param processed_features: The new feature set.
        :return: Whether the tag was matched by the pattern or not.
        """
        match = re.match(r'\[[^.|]+\.[^.]+(\.[^.]+)?\]', feature)
        if match is not None:
            feature = feature.replace('.', '][')
            processed_features.extend(re.findall('\[.*?\]', feature))
            return True

    def __check_main_minor(self, feature, processed_features):
        """
        Separates main and minor POS.
        Check format: [\<text>|<text>] second part from the first |
        :param feature: The feature which has to be processed.
        :param processed_features: The new feature set.
        :return: Whether the tag was matched by the pattern or not.
        """
        match = re.search(r'\|.*\]', feature)
        if match is not None:
            second_part = '[' + match.group(0)[1:]
            first_part = feature[0:feature.index(match.group(0))] + ']'
            processed_features.append(first_part)
            processed_features.append(second_part)
            return True
