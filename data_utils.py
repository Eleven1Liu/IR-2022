import nltk
import itertools
import logging

import pandas as pd
from spacy.lang.en import English

from evaluate import normalize_score, lcs


nlp = English()
sentencizer = nlp.add_pipe("sentencizer")
nltk.download('punkt')

PUNCTUATIONS = set([c for c in """!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"""])


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


def sample_nli_datasets(df, quieres, responses, q_true, r_true, q_indexes, r_indexes, threshold=0.5, train_test_split=0.2):
    """Sample data from NLI training.

    Args:
        df (pd.DataFrame): training data grouped by id
        queries (list): list of query sentences
        responses (list): list of responses sentences
        q_true (list): list of ground truth queries
        r_true (list): list of ground truth responses
        q_indexes (list): the `queries` offsets for each training instance
        r_indexes (list): the `responses` offsets for each training instance
        threshold (float, optional): thershold. Defaults to 0.5.
        train_test_split (float, optional): train test split. Defaults to 0.2.
    """
    data = { k: [] for k in ["id", "label", "text"] }
    for i in range(len(df)):
        qs, qe = q_indexes[i]
        selected_q = select_sentences(quieres[qs:qe], q_true[i], threshold)
        rs, re = r_indexes[i]
        selected_r = select_sentences(responses[rs:re], r_true[i], threshold)

        for q, r in itertools.product(selected_q, selected_r):
            data["id"].append(df["id"][i])
            data["label"].append(df["s"][i])
            data["text"].append(f"{q} [SEP] {r}") # hard-code SEP ..

    test_ids = list(df.sample(frac=train_test_split)["id"])
    df_all = pd.DataFrame(data)
    df_train = df_all[~df_all["id"].isin(test_ids)]
    df_test = df_all[df_all["id"].isin(test_ids)]
    return df_train, df_test


def select_sentences(sents, true_sents, threshold=0.5):
    """Select sentences for training the NLI model.

    Args:
        sents (list): list of sentences from `q` or `r`
        true_sents (list): list of sentences from `q'` or `r'`
        threshold (float, optional): thershold. Defaults to 0.5.

    Returns:
        List[str]: sentences with LCS scores > 0.5
    """
    selected_sents = list()
    for s in sents:
        toks = tokenize(s)
        for true_s in true_sents:
            _, true_all_sents = sentencize_docs([true_s])
            for s_ in true_all_sents:
                true_toks = tokenize(s_)
                score = normalize_score(lcs(toks, true_toks), toks, true_toks)
                if score > threshold:
                    selected_sents.append(s)
    return set(selected_sents)


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


def tokenize(text):
    """Tokenize text and remove punctuations."""
    tokens = nltk.word_tokenize(text)
    return [c for c in tokens if c not in PUNCTUATIONS]


def write_dataset(df, path):
    """Write dataset in LibMultiLabel format"""
    df.to_csv(path, sep='\t', index=False, header=False)
    logging.info("Write %d instances to %s." %(len(df), path))


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
