import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, tweets, labels, vocab, processor, max_length=128):
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
        self.vocab = vocab

    def __len__(self):
        return len(self.tweets)

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

    def ids_to_tokens(self, ids):
        """
        Converts a tensor of token IDs back into tokens using the dataset's vocabulary.

        Args:
            ids (torch.Tensor): Tensor containing token IDs.

        Returns:
            list: List of tokens corresponding to the IDs.
        """
        # Reverse the vocabulary: index -> token
        id_to_token = {idx: token for token, idx in self.vocab.items()}

        # Convert each ID to a token
        tokens = [id_to_token.get(id.item(), "<UNK>") for id in ids]
        return tokens
