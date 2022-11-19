import argparse
import torch
import yaml

from data_utils import load_data, sentencize_docs
from evaluate import group_ground_truth, score
from libmultilabel.common_utils import Timer, AttributeDict
from sentence_transformers import SentenceTransformer, util


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

    # Load model from HuggingFace Hub
    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_embeds = model.encode(q_all_sents)
    r_embeds = model.encode(r_all_sents)

    # Group q_embeds by q_ind
    num_instances = len(df_)
    scores = list()
    for i in range(num_instances):
        qs, qe = q_ind[i]
        rs, re = r_ind[i]
        cos_sim = util.cos_sim(q_embeds[qs:qe], r_embeds[rs:re])
        if df_["s"][i] == "DISAGREE":
            x = torch.argmin(cos_sim).item()
        else:
            x = torch.argmax(cos_sim).item()
        qi, ri = x // cos_sim.shape[1], x % cos_sim.shape[1]
        id_ = df_["id"][i]
        s = score(q_all_sents[qs+qi], r_all_sents[rs+ri],
                  q_true[id_], r_true[id_])
        scores.append(s)

    final_score = sum(scores) / (2*num_instances)
    print(f"Collect {len(scores)} scores.")
    print(f"Score: {final_score}")


if __name__ == '__main__':
    wall_time = Timer()
    main()
    print(f'Wall time: {wall_time.time():.2f} (s)')
