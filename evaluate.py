import logging

import nltk

nltk.download("punkt")
PUNCTUATIONS = set([c for c in """!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"""])


def group_ground_truth(df):
    """Group ground truth by id.

    Args:
        df (pd.DataFrame): Training data with `q'` and `r'`.

    Returns:
        pandas.core.series.Series: list of `q_true` and `r_true` grouped `id`.
    """
    groups = df.groupby(by=["id"])
    q_true = groups["q'"].apply(list).values.tolist()
    r_true = groups["r'"].apply(list).values.tolist()
    return q_true, r_true


def eval(queries, responses, q_indexes, r_indexes, selected_indexes, q_true=None, r_true=None):
    """Calculate prediction score for CosSimilarity and NLI.

    Args:
        queries (list): list of query sentences
        responses (list): list of responses sentences
        q_indexes (list): the `queries` offsets for each training instance
        r_indexes (list): the `responses` offsets for each training instance
        selected_indexes (List[tuple]): List of the most related/unrelated indexes
        q_true (list): list of ground truth queries, defaults to None
        r_true (list): list of ground truth responses, defaults to None
    """
    q_outs, r_outs, scores = list(), list(), list()
    N = len(q_indexes)
    for i in range(N):
        col_num = r_indexes[i][1]-r_indexes[i][0]
        # start + offset
        q_pred = queries[q_indexes[i][0] + selected_indexes[i] // col_num]
        r_pred = responses[r_indexes[i][0] + selected_indexes[i] % col_num]
        q_outs.append(q_pred)
        r_outs.append(r_pred)

        if q_true is not None and r_true is not None:
            s = score_one(q_pred, r_pred, q_true[i], r_true[i])
            scores.append(s)

    final_score = sum(scores) / (2*N)
    logging.info(f"Score: {final_score}")
    return q_outs, r_outs, scores


def eval_sentence_pairs(sentence_pairs, q_true, r_true):
    """Calculate prediction score for NLIClassifier.

    Args:
        sentence_pairs (list): list of predicted q and r split by [SEP].
        q_true (list): list of ground truth queries
        r_true (list): list of ground truth responses
    """
    q_outs, r_outs, scores = list(), list(), list()
    N = len(q_true)
    for i, text in enumerate(sentence_pairs):
        q_pred, r_pred = [t.strip() for t in text.split("[SEP]")]
        s = score_one(q_pred, r_pred, q_true[i], r_true[i])
        q_outs.append(q_pred)
        r_outs.append(r_pred)
        scores.append(s)

    final_score = sum(scores) / (2*N)
    logging.info(f"Score: {final_score}")
    return q_outs, r_outs, scores


def score_one(q_pred, r_pred, q_true: list, r_true: list):
    """Calculate the max LCS score of `j` possible answers.

    Args:
        q_pred (list): one predicted query
        r_pred (list): one predicted response
        q_true (list): list of ground truth queries
        r_true (list): list of ground truth qresponses
    Returns:
        int: max score of the prediction and the ground truth
    """
    q_pred_toks = tokenize(q_pred)
    r_pred_toks = tokenize(r_pred)

    score = 0
    for j in range(len(q_true)):
        q_true_toks = tokenize(q_true[j])
        r_true_toks = tokenize(r_true[j])
        q_score = normalize_score(
            lcs(q_pred_toks, q_true_toks), q_pred_toks, q_true_toks)
        r_score = normalize_score(
            lcs(r_pred_toks, r_true_toks), r_pred_toks, r_true_toks)
        score = max(score, q_score + r_score)
    return score


def normalize_score(score, text1, text2):
    """Normalize the score to a number between 0 and 1.

    Args:
        score (int): the LCS score of text1 and text2
        text1 (list): list of tokens
        text2 (list): list of tokens
    """
    if len(text1) == 0 or len(text2) == 0:
        return 0
    return score / (len(text1) + len(text2) - score)


def lcs(text1: list, text2: list) -> int:
    """LCS from the course PPT

    Args:
        text1 (list): list of tokens
        text2 (list): list of tokens

    Returns:
        int: length of LCS
    """
    if len(text2) > len(text1):
        text1, text2 = text2, text1
    lcs = [[0] * (len(text2) + 1) for _ in range(2)]
    for i in range(1, len(text1)+1):
        for j in range(1, len(text2)+1):
            if text1[i-1] == text2[j-1]:
                lcs[i % 2][j] = lcs[(i-1) % 2][j-1] + 1
            else:
                lcs[i % 2][j] = max(lcs[(i-1) % 2][j], lcs[i % 2][j-1])
    return lcs[len(text1) % 2][len(text2)]


def tokenize(text):
    """Tokenize text and remove punctuations. (for scoring)"""
    tokens = nltk.word_tokenize(text)
    return [c for c in tokens if c not in PUNCTUATIONS]
