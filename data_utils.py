import logging

import pandas as pd
from spacy.lang.en import English

nlp = English()
sentencizer = nlp.add_pipe("sentencizer")


def load_data(path, text_cols=None):
    """Load and preprocess tbrain data.

    Args:
        path (str): data path
        text_cols (list): text columns contains query and response
    """

    if text_cols is None:
        text_cols = ["q", "r", "q'", "r'"]

    df = pd.read_csv(path)
    df = df.fillna("")
    for col in text_cols:
        df[col] = df[col].apply(preprocess)
    return df


def preprocess(text):
    """Remove " from the begin and end."""
    return text.strip()[1:-1]


def sentencize(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def sentencize_docs(docs):
    """SBD for the collection"""
    # ind: map sentences to the original documents
    index, all_sents = list(), list()
    s = 0
    for d in docs:
        sents = sentencize(d)
        all_sents += sents
        index.append((s, s+len(sents)))
        s += len(sents)
    return index, all_sents


def write_results(queries, responses, output_path):
    """Output `q'` and `r'`.

    Args:
        queries (list): list of predicted `q'`
        responses (list): list of predicted `r'`
        output_path (str): output path
    """
    assert len(queries) == len(responses)
    df = pd.DataFrame({
        "q'": queries,
        "r'": responses
    })
    df.to_csv(output_path, index=False)
    logging.info("Write the %d results to %s." %
                 (len(queries), output_path))
