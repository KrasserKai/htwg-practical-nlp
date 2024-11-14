"""Tweet preprocessing module.

This module contains the TweetProcessor class which is used to preprocess tweets.

ASSIGNMENT-1:
Your job in this assignment is to implement the methods of this class.
Note that you will need to import several modules from the nltk library,
as well as from the Python standard library.
You can find the documentation for the nltk library here: https://www.nltk.org/
You can find the documentation for the Python standard library here: https://docs.python.org/3/library/
Your task is complete when all the tests in the test_preprocessing.py file pass.
You can check if the tests pass by running `make assignment-1` in the terminal.
You can follow the `TODO ASSIGNMENT-1` comments to find the places where you need to implement something.
"""

import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


class TweetProcessor:
    def __init__(self):
        self.tknzr = TweetTokenizer(
            preserve_case=False, reduce_len=True, strip_handles=True
        )

        self.stemmer = PorterStemmer()

    @staticmethod
    def remove_urls(tweet: str) -> str:
        """Remove urls from a tweet.

        Args:
            tweet (str): the input tweet

        Returns:
            str: the tweet without urls
        """

        url_pattern = r"http[s]?://\S+"

        return re.sub(url_pattern, "", tweet)

    @staticmethod
    def remove_hashtags(tweet: str) -> str:
        """Remove hashtags from a tweet.
        Only the hashtag symbol is removed, the word itself is kept.

        Args:
            tweet (str): the input tweet

        Returns:
            str: the tweet without hashtags symbols
        """

        hashtag_pattern = r"#"

        return re.sub(hashtag_pattern, "", tweet)

    def tokenize(self, tweet: str) -> list[str]:
        """Tokenizes a tweet using the nltk TweetTokenizer.
        This also lowercases the tweet, removes handles, and reduces the length of repeated characters.

        Args:
            tweet (str): the input tweet

        Returns:
            list[str]: the tokenized tweet
        """

        return self.tknzr.tokenize(tweet)

    @staticmethod
    def remove_stopwords(tokens: list[str]) -> list[str]:
        """Removes stopwords from a tweet.

        Only English stopwords are removed.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet without stopwords
        """

        stop_words = set(stopwords.words("english"))

        return [token for token in tokens if token not in stop_words]

    @staticmethod
    def remove_punctuation(tokens: list[str]) -> list[str]:
        """Removes standalone punctuation from a tweet but keeps emojis, mentions, hashtags, and common emoticons.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet without standalone punctuation
        """
        # Define a regex pattern to match standalone punctuation (but ignore emoticons and ellipses)
        punctuation_pattern = re.compile(r"^[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]+$")

        # Define a regex pattern for common emoticons and ellipses to preserve them
        emoticon_pattern = re.compile(
            r"(\.\.\.|:\)|:\(|:D|;D|:\*|:\||:P|;P|<3|:\]|:\[|:\}|\(:|\);|\^_\^|:-\))"
        )

        # Filter out tokens that are only punctuation, but keep tokens matching emoticons or ellipses
        return [
            token
            for token in tokens
            if not punctuation_pattern.match(token) or emoticon_pattern.match(token)
        ]

    def stem(self, tokens: list[str]) -> list[str]:
        """Stems the tokens of a tweet using the nltk PorterStemmer.

        Args:
            tokens (list[str]): the tokenized tweet

        Returns:
            list[str]: the tokenized tweet with stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]

    def process_tweet(self, tweet: str) -> list[str]:
        """Processes a tweet by removing urls, hashtags, stopwords, punctuation, and stemming the tokens.

        Args:
            tweet (str): the input tweet

        Returns:
            list[str]: the processed tweet
        """

        tweet_no_url = self.remove_urls(tweet)
        tweet_no_hashtag = self.remove_hashtags(tweet_no_url)
        tokens = self.tokenize(tweet_no_hashtag)
        tokens_no_stopword = self.remove_stopwords(tokens)
        tokens_no_punctation = self.remove_punctuation(tokens_no_stopword)
        tokens_stemmed = self.stem(tokens_no_punctation)

        return tokens_stemmed
