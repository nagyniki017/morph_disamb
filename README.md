# MorphDisamb

## A Hungarian morphological disambiguatior using recurrent and convolutional neural networks.

To analyse unknown words, [HFST](https://github.com/hfst/hfst) and the path for a transducer is required.

As a Hungarian transducer, [emMorph](https://github.com/dlt-rilmta/emMorph) can be used - please provide the path of the compiled transducer.

### Usage

#### Training
Training requires a corpus with the following format:
- empty lines separate the sentences
- other lines consist of tab-separated colums:
  - the first column holds the word
  - the last column holds the disambiguated analysis
  
To train a new (convolutional) model:
```
python main.py -t -C [--batch 64] [--epoch 128] [--directory corpus_directory] [--file corpus_file]
```

To continue the training of a saved (recurrent) model:
```
python main.py -t -R -l 2017-11-13-14-19 [--batch 64] [--epoch 128] [--directory corpus_directory] [--file corpus_file]
```

In case only the corpus directory is provided, each file within it will be handled as corpus file.

#### Evaluation
Evaluation requires a corpus with the same format as training.

To evaluate a fresh training:
```
python main.py -t -e -C [--batch 64] [--epoch 128] [--directory corpus_directory] [--file corpus_file] [-l 2017-11-16-15-54]
```

To evaluate a saved model:
```
python main.py -e -R -l 2017-11-13-14-19 [--directory corpus_directory] [--file corpus_file]
```

The output of the evaluation:
- writes the neural network loss and accuracy to standard output
- writes the disambiguation results into a file with the following properties:
  - file name format: disambiguated-<the build time of the network>.txt
  - sentences are separated by empty lines
  - the original word, and the expected and got analyses are written into the file (each in separate lines, and the analyses are indented)
  - at the end of the file, the correctly disambigguated word and sentence count and ratio is shown


#### Disambiguation
The source for disambiguation can be the standard input or a file.

The file can have the same format which was required for training and evaluation. Multiple columns aren't necessary, the file can hold only the words.

In case of use input, [quntoken](https://github.com/dlt-rilmta/quntoken) is required for tokenization. The user input has to be usual text without separating words into lines.

Disambiguation with file input:
```
python main.py -d -R -l 2017-11-13-14-19 --directory input_dir --file input_file
```

Disambiguation from standard input:
```
python main.py -d -R -l 2017-11-13-14-19
```
OR
```
cat input_file_path | python main.py -d -R -l 2017-11-13-14-19
```

## BibTex
```bibtex
@thesis{nagyn2017,
	author = {Nagy, Nikolett},
	title = {Hungarian morphological disambiguation using recurrent and convolutional neural networks},
	institution = {Budapest University of Technology and Economics},
	year = {2017}
}
```
