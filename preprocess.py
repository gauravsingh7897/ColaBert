import os
import wget
import pandas as pd
from sklearn.utils import shuffle


def load(train=True):

    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

    if not os.path.exists('./data/cola_public_1.1.zip'):
        wget.download(url, './data/cola_public_1.1.zip')

    mode = "train" if train else "dev"

    df = pd.read_csv(f'./data/cola_public/raw/in_domain_{mode}.tsv', delimiter='\t', header=None, names=[
                     'source', 'label', 'notes', 'sentence'])

    df = shuffle(df)
    df.reset_index(drop=True)

    sentences = df['sentence'].values
    labels = df['label'].values

    return sentences, labels