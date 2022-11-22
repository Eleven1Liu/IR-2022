import argparse
import logging
import sys

import yaml
from libmultilabel.common_utils import AttributeDict, Timer

from data_utils import load_data, sentencize_docs, write_results
from evaluate import group_ground_truth, eval
from rankers import NLI, CosSimilarity

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config/baseline.yml")
    args, _ = parser.parse_known_args()

    with open(args.config) as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    config = AttributeDict(config)

    # load training data
    df = load_data(config.train_path)
    q_true, r_true = group_ground_truth(df)
    df_ = df.drop_duplicates(subset=["id"], keep="first").reset_index()
    logging.info(f"Load {len(df_)} input samples.")

    # sbd
    q_ind, q_all_sents = sentencize_docs(df_["q"])
    r_ind, r_all_sents = sentencize_docs(df_["r"])
    logging.info(
        f"Load {len(q_all_sents)} q sentences and {len(r_all_sents)} r sentences.")

    # Load rankers
    ranker = getattr(sys.modules[__name__], config.ranker)()
    scores = ranker.predict(q_all_sents, r_all_sents, q_ind, r_ind)
    selected_indexes = ranker.rank(scores, df_["s"])

    q_outs, r_outs, scores = eval(
        q_all_sents, r_all_sents, q_true, r_true, q_ind, r_ind, selected_indexes)
    write_results(q_outs, r_outs, config.output_path)


if __name__ == '__main__':
    wall_time = Timer()
    main()
    print(f'Wall time: {wall_time.time():.2f} (s)')
