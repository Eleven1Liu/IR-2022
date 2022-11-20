import argparse
import yaml

from data_utils import load_data, sentencize_docs
from evaluate import group_ground_truth, score
from rankers import CosSimilarity, NLI
from libmultilabel.common_utils import Timer, AttributeDict


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default='config/baseline.yml')
    args, _ = parser.parse_known_args()

    with open(args.config) as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    config = AttributeDict(config)

    # load training data
    df = load_data(config.train_path)
    q_true, r_true = group_ground_truth(df)
    df_ = df.drop_duplicates(subset=["id"], keep='first').reset_index()
    print(f"Load {len(df_)} input samples.")

    # sbd
    q_ind, q_all_sents = sentencize_docs(df_["q"])
    r_ind, r_all_sents = sentencize_docs(df_["r"])
    print(f"Load {len(q_all_sents)} q sentences and {len(r_all_sents)} r sentences.")

    # Load rankers
    ranker = NLI()
    scores = ranker.predict(q_all_sents, r_all_sents, q_ind, r_ind)
    selected_indexes = ranker.rank(scores, df_["s"])

    num_instances = len(df_)
    scores = list()
    for i in range(num_instances):
        # start + offset
        qi = q_ind[i][0] + selected_indexes[i] // (r_ind[i][1]-r_ind[i][0])
        ri = r_ind[i][0] + selected_indexes[i] % (r_ind[i][1]-r_ind[i][0])
        id_ = df_["id"][i]
        s = score(q_all_sents[qi], r_all_sents[ri], q_true[id_], r_true[id_])
        scores.append(s)

    final_score = sum(scores) / (2*num_instances)
    print(f"Score: {final_score}")


if __name__ == '__main__':
    wall_time = Timer()
    main()
    print(f'Wall time: {wall_time.time():.2f} (s)')
