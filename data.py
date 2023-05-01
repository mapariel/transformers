import numpy as np
from nltk import word_tokenize, sent_tokenize
from torch.utils.data import Dataset

from utils import simple_cleansing, vectorize


class TransformerDataset(Dataset):
    """
    Custom Dataset. The x data is a large text (from Gutenberg https://www.gutenberg.org/)
    The text is divided into sequence of words of length n_sequence.
    """

    def __init__(self, text: str, n_sequence: int):
        self.n_sequence = n_sequence

        # Simple cleaning of the text, the tokens <SOS> and <EOS> are surrounding each sentence
        # and the text is split in a list of tokens (=words)
        text = simple_cleansing(text)
        text = ' '.join([' <SOS> ' + ' '.join(word_tokenize(s)) + ' <EOS> ' for s in sent_tokenize(text)])
        tokenized_text = text.split()

        # Creates the vocabulary : the list of unique words used in the text.
        # Tokens <SOS> and <EOS> are put at the beginning of the dictionary
        vocabulary = list(np.unique(tokenized_text))
        vocabulary.remove('<EOS>')
        vocabulary.remove('<SOS>')
        vocabulary = ['<SOS>', '<EOS>'] + vocabulary

        self.tokenized_text = tokenized_text
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.tokenized_text) // self.n_sequence

    def __getitem__(self, idx) -> np.ndarray:
        """
        Args:
            idx: The index of the sequence to be returned

        Returns:
            The indices of the sequence of tokens (=words) according to the vocabulary list
        """
        text = np.array(self.tokenized_text[idx * self.n_sequence:(1 + idx) * self.n_sequence + 1])
        vector = vectorize(text, self.vocabulary)
        return np.array(vector)
