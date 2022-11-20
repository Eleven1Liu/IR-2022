import itertools

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
            # qi, ri = x // cos_sim.shape[1], x % cos_sim.shape[1]
            # selected_indexes.append((qi, ri))
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
        for i in range(num_instances):
            qs, qe = q_indexes[i]
            rs, re = r_indexes[i]
            sent_pairs = [(q, r) for q, r in itertools.product(queries[qs:qe], responses[rs:re])]
            score = self.model.predict(
                sent_pairs, convert_to_numpy=False, convert_to_tensor=True)
            # labels = [self.label_names[score_max]
            #           for score_max in score.argmax(axis=1)]
            scores.append(score)
        return scores

    def rank(self, scores, states):
        selected_indexes = list()
        for i, score in enumerate(scores):
            # contradiction: 0, entailment: 1
            s = score[:, 1] if states[i] == "AGREE" else score[:, 0]
            x = torch.argmax(s).item()
            selected_indexes.append(x)
        return selected_indexes
