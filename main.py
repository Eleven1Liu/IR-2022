import argparse
import logging
import sys

import yaml
from libmultilabel.common_utils import AttributeDict, Timer

from data_utils import convert_nli_test, load_data, sample_nli_datasets, sentencize_docs, write_dataset, write_results
from evaluate import group_ground_truth, eval, eval_sentence_pairs
from rankers import NLI, CosSimilarity, NLIClassifier

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config/baseline.yml")
    args, _ = parser.parse_known_args()

    with open(args.config) as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    config = AttributeDict(config)

    # TODO simulate when there's only testing data for all rankers
    # load training data
    df = load_data(config.train_path)
    q_true, r_true = group_ground_truth(df)
    df_ = df.drop_duplicates(subset=["id"], keep="first").reset_index()
    logging.info(f"Load {len(df_)} input samples.")

    # sbd
    q_ind, query_sents = sentencize_docs(df_["q"])
    r_ind, respo_sents = sentencize_docs(df_["r"])
    logging.info(
        f"Load {len(query_sents)} query(q) sentences and {len(respo_sents)} response(r) sentences.")

    # Rankers
    ranker = getattr(sys.modules[__name__], config.ranker)()
    if config.ranker == "NLIClassifier":
        df_test = ranker.load_test_data(config.test_path)
        scores = ranker.load_prediction(config.predict_path)
        selected_indexes = ranker.rank(df_test, scores)
        sentence_pairs = df_test.loc[selected_indexes]["text"]
        q_true, r_true = group_ground_truth(df[df["id"].isin(df_test["id"])])
        q_outs, r_outs, scores = eval_sentence_pairs(
            sentence_pairs, q_true, r_true)
    else:
        scores = ranker.predict(query_sents, respo_sents, q_ind, r_ind)
        selected_indexes = ranker.rank(scores, df_["s"])
        q_outs, r_outs, scores = eval(
            query_sents, respo_sents, q_true, r_true, q_ind, r_ind, selected_indexes)
    write_results(q_outs, r_outs, config.output_path)


if __name__ == '__main__':
    wall_time = Timer()
    main()
    print(f'Wall time: {wall_time.time():.2f} (s)')
