import torch.nn as nn


def build_vocab(tweets, processor):
    """
    Dynamically build a vocabulary from the dataset.

    Args:
        tweets (list of str): List of tweet strings.

    Returns:
        dict: Token-to-index mapping.
    """
    from collections import Counter

    # Tokenize all tweets and count token frequencies
    counter = Counter()
    for tweet in tweets:
        tokens = processor(tweet)
        counter.update(tokens)

    # Assign a unique index to each token, starting from 2
    # Reserve 0 for <PAD> and 1 for <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token in counter:
        vocab[token] = len(vocab)

    return vocab


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            embed_dim (int): Dimension of the embedding vectors.
        """
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )  # Embedding layer with padding_idx
        self.fc = nn.Linear(embedding_dim, 1)  # Fully connected layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, 1).
        """
        # Embedding layer
        embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embed_dim)

        # Average pooling over the sequence length, ignoring padding (handled automatically by padding_idx)
        pooled_embedding = embedded.mean(dim=1)  # Shape: (batch_size, embed_dim)

        # Fully connected layer
        logits = self.fc(pooled_embedding)  # Shape: (batch_size, 1)

        # Sigmoid activation for binary classification
        output = self.sigmoid(logits)  # Shape: (batch_size, 1)

        return output
