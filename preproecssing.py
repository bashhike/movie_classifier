#! /bin/python3

import numpy as np
from torchtext.legacy import data
import spacy
import re
import random


class PreprocessData(object):
    def __init__(self, allowed_genre):
        super(PreprocessData, self).__init__()
        self.genre_dict = self._build_genre_dict(allowed_genre)
        self.inverse_genre_dict = self._build_inverse_genre_dict(allowed_genre)

        # Spacy related variables.
        self.tokenizer = spacy.load('en_core_web_sm')
        self.stop_words = spacy.lang.en.STOP_WORDS

        # Fields from the data.
        self.text = data.Field(sequential=True, lower=True, tokenize=self.spacy_token,
                               stop_words=self.stop_words, eos_token='EOS',
                               include_lengths=True)
        self.label = data.LabelField(sequential=False, use_vocab=False,
                                     preprocessing=self.label_processing, pad_token=None,
                                     unk_token=None)

    def spacy_token(self, x):
        """
        Tokenize a given sentence using spacy's tokenizer
        :param x: Sentence
        :return: list of words/tokens
        """
        x = re.sub(r'[^a-zA-Z\s]', '', x)
        x = re.sub(r'[\n]', ' ', x)
        return [tok.text for tok in self.tokenizer.tokenizer(x)]

    def _build_genre_dict(self, allowed_genre):
        """
        Build a dictionary mapping a genre to a numerical entity.
        :param allowed_genre: A list of all the genres to focus on.
        :return: Dictionary containing genre and it's corresponding numerical entity.
        """
        genre_dict = dict()
        counter = 0
        for ele in allowed_genre:
            genre_dict[ele] = counter
            counter += 1
        return genre_dict

    def _build_inverse_genre_dict(self, allowed_genre):
        """
        Build a dictionary to map numbers to genres.
        :param allowed_genre: A list of genres to focus on.
        :return: Dictionary with numbers as keys and genre as values.
        """
        inverse_genre_dict = dict()
        counter = 0
        for ele in allowed_genre:
            inverse_genre_dict[counter] = ele
            counter += 1
        return inverse_genre_dict

    def label_processing(self, label):
        """
        Generate a one hot encoding for the label.
        :param label: genre name.
        :return: numpy array with a one hot representation of label.
        """
        ret = np.zeros(len(self.genre_dict), dtype=np.float32)
        ret[self.genre_dict[label]] = 1.0
        return ret

    def get_iterators(self, filename, batch_size, seed, device):
        """
        Generate iterators for the training and validation data.
        :param filename: path to the training data file.
        :param batch_size: batch size to use in the iterators.
        :param seed: random seed to keep the random splits consistent.
        :param device: device to store the tensors at.
        :return: Iterators for training and validation samples.
        """
        data_field = [(None, None), ("overview", self.text), ("genre", self.label)]
        dataset = data.TabularDataset(path=filename, format='csv', fields=data_field,
                                      skip_header=True)
        train_data, val_data = dataset.split(split_ratio=0.8, random_state=random.seed(seed))

        self.text.build_vocab(train_data)
        self.text.vocab.load_vectors('glove.6B.50d')

        train_iterator, valid_iterator = data.BucketIterator.splits(
            (train_data, val_data),
            batch_size=batch_size,
            sort_key=lambda x: len(x.overview),
            sort_within_batch=True,
            device=device)

        return train_iterator, valid_iterator

    def get_text(self):
        """
        Helper function to get the text variable.
        :return: text field object.
        """
        return self.text

    def get_genre_dict(self, inverse=False):
        """
        Helper function for fetching the genre dictionary.
        :param inverse: Return the inverse dict, with counters as keys.
        :return: Dictionary of genre and counters.
        """
        if inverse:
            return self.inverse_genre_dict
        else:
            return self.genre_dict

