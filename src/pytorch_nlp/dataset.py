from collections import Counter

import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, tweets, labels, processor, vocab=None, max_length=128):
        """
        Args:
            tweets (list of str): List of tweet strings.
            labels (list of int): Corresponding labels for each tweet.
            vocab (dict, optional): Mapping from tokens to numerical indices. If None, will be built dynamically.
            max_length (int): Maximum length for tokenized sequences.
        """
        self.tweets = tweets
        self.labels = labels
        self.processor = processor
        self.max_length = max_length
        self.vocab = self.build_vocab() if vocab is None else vocab

    def __len__(self):
        return len(self.tweets)

    def build_vocab(self):
        """
        Dynamically build a vocabulary from the dataset.

        Args:
            tweets (list of str): List of tweet strings.

        Returns:
            dict: Token-to-index mapping.
        """

        # Tokenize all tweets and count token frequencies
        counter = Counter()
        for tweet in self.tweets:
            tokens = self.processor(tweet)
            counter.update(tokens)

        # Assign a unique index to each token, starting from 2
        # Reserve 0 for <PAD> and 1 for <UNK>
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for token in counter:
            vocab[token] = len(vocab)

        return vocab

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]

        tokens = self.processor(tweet)

        # Convert tokens to indices using the vocabulary
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Pad or truncate the sequence
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            token_ids += [self.vocab["<PAD>"]] * (self.max_length - len(token_ids))

        # Convert to PyTorch tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_ids, label_tensor
