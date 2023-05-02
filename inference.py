import csv
import re
from typing import List

import torch

import model
from utils import vectorize

def generate_text(
        model : model.Model,
        vocabulary: List[str],
        start: str,
        length: int = 30,
        n_sequence: int = 50) -> str :
    """
    Generates a text, inferred from a starting sequence by a transformer model

    Args:
        model: The deep neural model based on self attention and transformer layers
        vocabulary: List of words corresponding to the indices used bt the model
        start: Start text fed to the model. A list of words separated by a space
        length: The minimum number of words output by the inference
        n_sequence: Maximum number of words used to infer the next word

    Returns:
        The sentence with at least length words, finishing with a point mark.
    """
    tokens = start.split()
    next_token = ''
    with torch.no_grad():
        while len(tokens) < length or next_token != '<EOS>':
            toks = tokens[-n_sequence:]
            vector = vectorize(toks, vocabulary)
            vector = torch.Tensor(vector).unsqueeze(0).to(torch.long)
            y_hat = model(vector)
            weights = torch.nn.functional.softmax(y_hat[0, -1], dim=0)
            # next_token = vocabulary[ torch.argmax(weights) ]
            next_token = vocabulary[torch.multinomial(weights, 1)]

            tokens.append(next_token)
            if next_token == '<EOS>':
                tokens.append('<SOS>')

    text = ''
    is_capitalized = False
    for token in tokens:
        if token == '<SOS>':
            is_capitalized = True
        elif token != '<EOS>':
            if is_capitalized:
                token = token.capitalize()
                is_capitalized = False
            text = text + ' ' + token

    text = '.'.join(re.split(r'[.]{2,}', text))  # Change consecutive dots (.....) by only one dot.
    return text


if __name__ == '__main__':
    with open('outputs/vocabulary.csv', newline='') as csvfile:
        dico_reader = csv.reader(csvfile, delimiter=' ')
        vocabulary = [row[0] for row in dico_reader]

    model = torch.load('outputs/model.pt')
    model.eval()

    start = '<SOS>  he was '
    print(generate_text(model, vocabulary, start, length=35))
