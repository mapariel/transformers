import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, n_features, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_features, 2) * (-math.log(10000.0) / n_features))
        pe = torch.zeros(1, 1, max_len, n_features)
        pe[0, 0, :, 0::2] = torch.sin(position * div_term)
        pe[0, 0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: shape : n_batch, n_heads, n_sequence, n_features
        Returns:
            shape : n_batch, n_heads, n_sequence, n_features
        """
        x = x + self.pe[:, :, :x.size(2)]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, n_vocabulary: int, n_heads: int, n_features: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n_vocabulary, n_features) for _ in range(n_heads)])

    def forward(self, seq):
        """
        returns batch_size, n_heads, n_sequence, n_features
        """
        return torch.stack([embd(seq) for embd in self.embeddings], dim=1)


class Attention(nn.Module):
    def __init__(self, n_heads: int, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.q_mat = nn.ModuleList([nn.Linear(n_features, n_features) for _ in range(n_heads)])
        self.k_mat = nn.ModuleList([nn.Linear(n_features, n_features) for _ in range(n_heads)])
        self.v_mat = nn.ModuleList([nn.Linear(n_features, n_features) for _ in range(n_heads)])
        self.n_head = n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape : n_batch, n_heads, n_sequence, n_features
        Returns:
            shape : n_batch, n_heads, n_sequence, n_features
        """
        n_sequence = x.shape[-2]
        x = torch.swapaxes(x, 0, 1)

        keys = [self.k_mat[n](x[n]) for n in range(self.n_head)]
        queries = [self.q_mat[n](x[n]) for n in range(self.n_head)]
        values = [self.v_mat[n](x[n]) for n in range(self.n_head)]

        queries = torch.stack(queries, dim=1)
        keys = torch.stack(keys, dim=1).transpose(2, 3)
        values = torch.stack(values, dim=1)

        weights = torch.matmul(queries, keys)

        mask = torch.triu(torch.ones(n_sequence, n_sequence), 1).to(torch.bool)
        mask = mask.to(weights.device)

        weights = weights.masked_fill_(mask, -torch.inf)

        weights = weights / math.sqrt(self.n_features)
        weights = nn.Softmax(dim=-1)(weights)

        return torch.matmul(weights, values)


class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int , n_features: int):
        super().__init__()
        self.attention = Attention(n_heads, n_features)
        self.norm1 = nn.BatchNorm2d(n_heads)
        self.dense = nn.Linear(n_features, n_features)
        self.activation = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape : n_batch, n_heads, n_sequence, n_features
        Returns:
            shape n_batch, n_heads, n_sequence, n_features
        """
        y = self.attention(x)
        y = y + x
        y = self.norm1(y)
        z = self.dense(y)
        z = self.activation(z)
        z = z + y
        z = self.norm2(z)
        return z


class Model(nn.Module):
    def __init__(self, n_vocabulary: int, n_heads: int = 4, n_features:int = 256):
        """
        Transformer Deep Neural Network model.
        Model is described in https://web.stanford.edu/~jurafsky/slp3/ ch10. Transformers and Pretrained Language Models

        Args:
            n_vocabulary: the number of words
            n_heads: the number of heads for the transformer
            n_features: the length of the word embedding vectors
        """
        super().__init__()
        self.embeddings = Embeddings(n_vocabulary, n_heads, n_features)
        self.pe = PositionalEncoding(n_features, max_len=100)
        self.transformer_block = TransformerBlock(n_heads, n_features)
        self.dense = nn.Linear(n_features * n_heads, n_vocabulary)
        self.activation = nn.ReLU()

    def forward(self, x):
        y = self.embeddings(x)
        y = self.pe(y)
        y = self.transformer_block(y)
        y = y.transpose(1, 2)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        y = self.dense(y)
        return y
