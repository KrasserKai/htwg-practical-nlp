"""feature module for NLP.

This module contains the CountVectorizer class for NLP tasks.
"""

from collections import defaultdict

import numpy as np


class CountVectorizer:
    """CountVectorizer class for NLP tasks.

    This class implements a CountVectorizer for NLP tasks.
    It can be used to build a feature vector for a given tweet.

    Attributes:
        word_frequencies (defaultdict[tuple[str, int], int]): a dictionary of word frequencies by class
    """

    def __init__(self):
        """Initializes the CountVectorizer.

        The `word_frequencies` attribute is a dictionary of word frequencies by class.

        Makes use of [Python's defaultdict](https://docs.python.org/3/library/collections.html#collections.defaultdict) to initialize the dictionary with 0 for each key.
        This means that if a key is not found in the dictionary, the value is 0 and no KeyError exception is raised.

        """
        self.word_frequencies = defaultdict(int)

    def build_word_frequencies(
        self, tweets: list[list[str]], labels: np.ndarray
    ) -> None:
        """Builds a dictionary of word frequencies by counting the number of occurrences of each word in each class.

        The key is a tuple of the word and the class, the value is the frequency of the word in the class.
        For example, the key ('happi', 1) refers to the word 'happi' in the positive class,
        and the value it holds is the number of times the word 'happi' occurs in the positive class.

        Example:
            >>> vec = CountVectorizer()
            >>> vec.build_word_frequencies(tweets, labels)
            >>> vec.word_frequencies[('happi', 1)]
            42

        Args:
            tweets (list[list[str]]): a list of tokenized tweets
            labels (list[str]): a list of corresponding class labels

        """
        for tweet, label in zip(tweets, labels):
            for word in tweet:
                self.word_frequencies[
                    (word, label.item() if isinstance(label, np.ndarray) else label)
                ] += 1

    def get_features(self, tweet: list[str]) -> np.ndarray:
        """Returns a feature vector for a given tweet.

        The feature vector is a row vector of the form `[1, pos, neg]`, where `pos` and `neg` are the sum of the frequencies of the words in the tweet in the positive and negative class, respectively.

        Args:
            tweet (list[str]): a tokenized tweet

        Returns:
            np.ndarray: the feature vector for the tweet as a row vector
        """

        pos = 0  # Sum of frequencies in the positive class
        neg = 0  # Sum of frequencies in the negative class

        for word in tweet:
            pos += self.word_frequencies.get(
                (word, 1), 0
            )  # Frequency of the word in the positive class
            neg += self.word_frequencies.get(
                (word, 0), 0
            )  # Frequency of the word in the negative class

        # Feature vector: [1, pos, neg]
        return np.array([1, pos, neg])
