# Datasets for your training, DO NOT MODIFY!!!
# Read the codes carefully to implment your trainers properly.
import os
from typing import List, Union
from itertools import chain
import csv

from bpe import BytePairEncoding

import sentencepiece as spm

class ParagraphDataset(object):
    """ Paragraph Dataset
    We use same IMDB datset for pretraining.
    This dataset stores reviews and returns one of them when it called.
    You can use a returned review as a paragraph to sample a sentence.
    """
    def __init__(self, file_path: str) -> None:
        self._processor = spm.SentencePieceProcessor()
        self._processor.load(os.path.join('sentencepiece', 'imdb.model'))

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            self._rows = [row[1:] for row in reader]

    @property
    def token_num(self):
        # The number of tokens. Use this to sample a word
        return len(self._processor)

    def __len__(self) -> int:
        # The number of the paragraphs
        return len(self._rows)

    def __getitem__(self, index: int) -> List[List[str]]:
        # Return one of the paragraphs
        return list(map(self._processor.EncodeAsIds, self._rows[index]))

class IMDBdataset(object):
    """  IMDB Review Dataset
    This dataset stores reviews and returns one of them and its sentiment label when it called.
    """
    def __init__(self, file_path: str) -> None:
        self._processor = spm.SentencePieceProcessor()
        self._processor.load(os.path.join('sentencepiece', 'imdb.model')) 

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            self._rows = [row for row in reader]

    @property
    def token_num(self):
        # The number of tokens. This includes <PAD> token
        return len(self._processor)

    def __len__(self):
        # The number of reviews
        return len(self._rows)

    def __getitem__(self, index):
        # Retrun one of the review
        row = self._rows[index]

        label = True if row[0] == 'positive' else False
        sentences = list(map(self._processor.EncodeAsIds, row[1:]))
        sentences = [BytePairEncoding.CLS_token_idx] + list(chain(*sentences)) + [BytePairEncoding.SEP_token_idx]

        return sentences, label

#############################################
# Testing functions below.                  #
#############################################

def test_paragraph_dataset():
    print("======Paragraph Database Test======")
    dataset = ParagraphDataset('data/imdb_train.csv')
    print("Paragraph Format: {}".format(dataset[0]))
    print("The number of paragraphs: {}".format(len(dataset)))
    print("Maximum sentence length: {}".format(max(max(len(sent) for sent in paragraph) for paragraph in dataset)))

def test_IMDB_dataset():
    print("======IMDB Database Test======")
    dataset = IMDBdataset('data/imdb_train.csv')
    print("Review Format: {}".format(dataset[0]))
    print("The number of reviews: {}".format(len(dataset)))
    print("Maximum review length: {}".format(max(len(data[0]) for data in dataset)))

if __name__ == "__main__":
    test_paragraph_dataset()
    test_IMDB_dataset()
