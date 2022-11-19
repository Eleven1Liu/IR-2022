import nltk
nltk.download('punkt')


def group_ground_truth(df):
    """Group ground truth by id.

    Args:
        df (pd.DataFrame): Training data with `q'` and `r'`.

    Returns:
        pandas.core.series.Series: list of `q_true` and `r_true` grouped `id`.
    """
    groups = df.groupby(by=["id"])
    q_true = groups["q'"].apply(list)
    r_true = groups["r'"].apply(list)

    return q_true, r_true


def score(q_pred, r_pred, q_true: list, r_true: list):
    """Calculate the max LCS score of `j` possible answers.

    Args:
        q_pred (_type_): one predicted query
        r_pred (_type_): one predicted response
        q_true (list): list of ground truth queries
        r_true (list): list of ground truth qresponses
    Returns:
        int: max score of the prediction and the ground truth
    """
    q_pred_toks = nltk.word_tokenize(q_pred)
    r_pred_toks = nltk.word_tokenize(r_pred)

    score = 0
    for j in range(len(q_true)):
        q_true_toks = nltk.word_tokenize(q_true[j])
        r_true_toks = nltk.word_tokenize(r_true[j])
        q_score = lcs(q_pred_toks, q_true_toks) / len(set(q_pred_toks + q_true_toks))
        r_score = lcs(r_pred_toks, r_true_toks) / len(set(r_pred_toks + r_true_toks))
        score = max(score, q_score + r_score)
    return score


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
