import itertools
import logging

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from evaluate import score


class Ranker():
    def __init__(self) -> None:
        pass

    def evaluate():
        pass


class CosSimilarity(Ranker):
    def __init__(self, model_name=None) -> None:
        super().__init__()
        self.model_name = model_name or 'all-MiniLM-L6-v2'
        self._load_model()

    def _load_model(self):
        self.model = SentenceTransformer(self.model_name)

    def predict(self, queries, responses, q_indexes, r_indexes):
        """

        Args:
            queries (_type_): _description_
            responses (_type_): _description_
            q_indexes (_type_): _description_
            r_indexes (_type_): _description_
        """
        q_embeds = self.model.encode(queries)
        r_embeds = self.model.encode(responses)

        num_instances = len(q_indexes)
        logging.info(f'Load {num_instances} instances.')

        scores = list()
        for i in range(num_instances):
            qs, qe = q_indexes[i]
            rs, re = r_indexes[i]
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
            qi, ri = x // cos_sim.shape[1], x % cos_sim.shape[1]
            selected_indexes.append((qi, ri))
        return selected_indexes


class NLI(Ranker):
    def __init__(self, encoder_name=None) -> None:
        super().__init__()
        self.encoder_name = encoder_name or 'cross-encoder/nli-deberta-v3-base'
        self.label_names = ['contradiction', 'entailment', 'neutral']
        self._load_model()

    def _load_model(self):
        self.model = CrossEncoder(self.encoder_name)

    def predict(self, queries, responses):
        sent_pairs = [(q,r) for q, r in itertools.product(queries, responses)]
        scores = self.model.predict(sent_pairs)
        # labels = [self.label_names[score_max] for score_max in scores.argmax(axis=1)]
        return scores

    def rank(self, scores, states):
        pass
