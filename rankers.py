import collections
import itertools
import logging

import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer


class Ranker():
    def __init__(self) -> None:
        pass

    def evaluate():
        pass


class CosSimilarity(Ranker):
    def __init__(self, model_name=None) -> None:
        super().__init__()
        self.model_name = model_name or 'all-MiniLM-L6-v2'  # 'multi-qa-MiniLM-L6-cos-v1'
        self._load_model()

    def _load_model(self):
        self.model = SentenceTransformer(self.model_name)

    def predict(self, queries, responses, q_indexes, r_indexes):
        """Cosine similarity scores of (q, r) sentence pairs.

        Args:
            queries (list): list of query sentences
            responses (list): list of responses sentences
            q_indexes (list): the `queries` offsets for each training instance
            r_indexes (list): the `responses` offsets for each training instance

        Returns:
            List[torch.tensor]: List of len(q)*len(r) cosine similarity scores.
        """
        q_embeds = self.model.encode(queries)
        r_embeds = self.model.encode(responses)
        num_instances = len(q_indexes)

        scores = list()
        for i in tqdm(range(num_instances)):
            qs, qe = q_indexes[i]
            rs, re = r_indexes[i]
            # dot_sim = util.dot_score(q_embeds[qs:qe], r_embeds[rs:re])
            cos_sim = util.cos_sim(q_embeds[qs:qe], r_embeds[rs:re])
            scores.append(cos_sim)
        return scores

    def rank(self, scores, states):
        """Return the top-1 related/unrelated (q_ind,r_ind) of each instance.

        Args:
            scores (list): Cosine similarity scores.
            states (list): AGREE/DISAGREE.

        Returns:
            List[tuple]: List of the most related/unrelated indexes.
        """
        selected_indexes = list()
        for i, cos_sim in enumerate(scores):
            if states[i] == "AGREE":
                x = torch.argmax(cos_sim).item()
            else:
                x = torch.argmin(cos_sim).item()
            selected_indexes.append(x)
        return selected_indexes


class NLI(Ranker):
    def __init__(self, encoder_name=None) -> None:
        super().__init__()
        self.encoder_name = encoder_name or 'cross-encoder/nli-deberta-v3-base'
        self.label_names = ['contradiction', 'entailment', 'neutral']
        self._load_model()

    def _load_model(self):
        self.model = CrossEncoder(self.encoder_name)

    def predict(self, queries, responses, q_indexes, r_indexes):
        """Contradiction, entailment, and neutral scores of (q, r) sentence pairs.
         TODO batch eval

        Args:
            queries (list): list of query sentences
            responses (list): list of responses sentences
            q_indexes (list): the `queries` offsets for each training instance
            r_indexes (list): the `responses` offsets for each training instance

        Returns:
            numpy.ndarray: contradiction, entailment, and neutral scores for query-response pairs.
        """
        num_instances = len(q_indexes)
        scores = list()
        for i in tqdm(range(num_instances)):
            qs, qe = q_indexes[i]
            rs, re = r_indexes[i]
            sent_pairs = [(q, r) for q, r in itertools.product(queries[qs:qe], responses[rs:re])]
            score = self.model.predict(
                sent_pairs, convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False)
            # labels = [self.label_names[score_max]
            #           for score_max in score.argmax(axis=1)]
            scores.append(score)
        return scores

    def rank(self, scores, states):
        selected_indexes = list()
        for i, score in enumerate(scores):
            # contradiction: 0, entailment: 1, neural: 2
            # s = score[:, 1] if states[i] == "AGREE" else score[:, 0]
            if states[i] == "AGREE":
                entail_max = torch.argmax(score[:, 1])
                neural_max = torch.argmax(score[:, 2])
                s = score[:, 1] if entail_max > neural_max else score[:, 2]
            else:
                s = score[:, 0]
            x = torch.argmax(s).item()
            selected_indexes.append(x)
        return selected_indexes

    # def rank_topk(self, scores, states):
    # TBD


class NLIClassifier(Ranker):
    def __init__(self):
        super().__init__()

    def predict(self):
        pass

    def rank(self, df, scores):
        # Select label scores based on given "s" (AGREE/DISAGREE)
        df["preds"] = [scores[df["s"][i]][i] for i in range(len(df))]
        selected_indexes = df.groupby(["id"])["preds"].idxmax()
        return selected_indexes

    def load_prediction(self, pred_path):
        """Load label scores for sentence pairs.

        Args:
            pred_path (str): path to the prediction in
                "label1:score1 label2:score2" format.

        Returns:
            dict: `AGREE` and `DISAGREE` scores for query-response pairs.
        """
        with open(pred_path, "r") as f:
            lines = f.read().splitlines()

        scores = collections.defaultdict(list)
        for line in lines:
            label_scores = [l.split(":") for l in line.split(" ")]
            for label, score in label_scores:
                scores[label].append(float(score))
        logging.info("Load %d prediction scores." % (len(lines)))
        return scores

    def load_test_data(self, test_path):
        """Load test data with columns `id`, `s`, and `text` (sentence pairs split by [SEP])."""
        df_test = pd.read_csv(test_path, sep="\t", header=None)
        df_test.columns = ["id", "s", "text"]
        logging.info("Load %d test instances." %(len(df_test)))
        return df_test


class BART:
    def __init__(self):
        super().__init__()
        self.encoder_name = 'facebook/bart-large-cnn'
        self._load_model()

    def _load_model(self):
        self.tokenizer = BartTokenizer.from_pretrained(self.encoder_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.encoder_name)

    def predict(self, queries, responses, q_indexes, r_indexes):
        num_instances = len(q_indexes)
        q_outs, r_outs = list(), list()
        for i in tqdm(range(num_instances)):
            qs, qe = q_indexes[i]
            rs, re = r_indexes[i]
            q_outs.append(self.summarize(queries[qs:qe]))
            r_outs.append(self.summarize(responses[rs:re]))
        return q_outs, r_outs

    def summarize(self, sentences, limit=20):
        document = (" ".join(sentences[:limit]))
        inputs = self.tokenizer([document],
                        max_length=1024, return_tensors="pt")

        # Generate Summary
        summary_ids = self.model.generate(
            inputs["input_ids"], num_beams=2, min_length=0, max_length=20)

        return self.tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
