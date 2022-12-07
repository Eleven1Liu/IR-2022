import collections
import itertools
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator
from beir.generation.models import QGenModel
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from data_utils import write_qgen_corpus


class Ranker():
    def __init__(self) -> None:
        pass

    def predict():
        pass


class CosSimilarity(Ranker):
    def __init__(self, model_name=None, **kwargs) -> None:
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
    def __init__(self, encoder_name=None, **kwargs) -> None:
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
    def __init__(self, **kwargs):
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
    def __init__(self, sentence_limit=20, summary_max_length=50, rerank_threshold=0.5, gpl_checkpoint_path=None, **kwargs):
        super().__init__()
        self.encoder_name = "facebook/bart-large-cnn"  # sshleifer/distilbart-cnn-12-6
        self.cross_encoder_name = "cross-encoder/stsb-distilroberta-base"
        self.gpl_checkpoint_path = gpl_checkpoint_path
        self.sentence_limit = sentence_limit # sentence num for summarization after SBD
        self.summary_max_length = summary_max_length
        self.rerank_threshold = rerank_threshold

        self._init_device()
        self._load_model()

    def _init_device(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}")

    def _load_model(self):
        self.tokenizer = BartTokenizer.from_pretrained(self.encoder_name)
        self.model = BartForConditionalGeneration.from_pretrained(
            self.encoder_name).to(self.device)
        self.cross_model = CrossEncoder(self.cross_encoder_name)
        self.reranker = SentenceTransformer(self.gpl_checkpoint_path)

    def predict(self, queries, responses, q_indexes, r_indexes, rerank_sentences=True):
        num_instances = len(q_indexes)
        q_outs, r_outs = list(), list()
        for i in tqdm(range(num_instances)):
            qs, qe = q_indexes[i]
            rs, re = r_indexes[i]
            q_preds = self.summarize(queries[qs:qe])
            r_preds = self.summarize(responses[rs:re])

            if rerank_sentences and self.gpl_checkpoint_path is None:
                q_preds = self.rerank_sentences(q_preds, queries[qs:qe])
                r_preds = self.rerank_sentences(r_preds, responses[rs:re])
            else:
                q_preds = self.rerank_gpl(q_preds, queries[qs:qe])
                r_preds = self.rerank_gpl(r_preds, responses[rs:re])

            q_outs.append(q_preds)
            r_outs.append(r_preds)
        return q_outs, r_outs

    def summarize(self, sentences):
        document = (" ".join(sentences[:self.sentence_limit]))
        inputs = self.tokenizer([document],
                                max_length=1024, return_tensors="pt").to(self.device)
        # generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"], num_beams=2,
            min_length=0, max_length=self.summary_max_length)

        return self.tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def rerank_sentences(self, summary, sentences):
        # sentence 1 = summary of passage, sentence 2s = sentences in the passage
        sentence_combinations = [[summary, sent] for sent in sentences]
        similarity_scores = self.cross_model.predict(
            sentence_combinations, show_progress_bar=False)
        indexes = np.where(similarity_scores > self.rerank_threshold)[0]
        return " ".join([sentences[ind] for ind in indexes])

    def rerank_gpl(self, summary, sentences):
        summary_embedding = self.reranker.encode(summary, show_progress_bar=False)
        similarity_scores = list()
        for sent in sentences:
            sent_embedding = self.reranker.encode(
                sent, show_progress_bar=False)
            cos_sim = util.cos_sim(sent_embedding, summary_embedding)
            similarity_scores.append(cos_sim.item())
        similarity_scores = np.array(similarity_scores)
        indexes = np.where(similarity_scores > self.rerank_threshold)[0]
        return " ".join([sentences[ind] for ind in indexes])

class GPL:
    """T5QGen + DenseRetrievalExactSearch
    - GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval
        (https://aclanthology.org/2022.naacl-main.168.pdf)
    """

    def __init__(self, gpl_checkpoint_path, ques_per_passage=30, qgen_dir='output/qgen', **kwargs):
        super().__init__()
        # QGen
        self.qgen_model_name = "BeIR/query-gen-msmarco-t5-base-v1"
        self.qgen_dir = qgen_dir
        self.qgen_corpus = dict()
        self.ques_per_passage = ques_per_passage

        # GPL
        self.gpl_checkpoint_path = gpl_checkpoint_path
        self._load_model()

        os.makedirs(self.qgen_dir, exist_ok=True)

    def _load_model(self):
        self.generator = QueryGenerator(model=QGenModel(self.qgen_model_name))
        # Dense Retrieval Exact Search
        model = SentenceTransformer(self.gpl_checkpoint_path)
        sbert = models.SentenceBERT(sep=" ")
        sbert.q_model = model
        sbert.doc_model = model
        self.dense_retriever = DRES(sbert, batch_size=16, show_progress_bar=False)

    def _load_qgen_corpus(self, passages, indexes, prefix):
        """Load corpus for question generation. Write corpus to `qgen_dir/{prefix}corpus.jsonl`
        if the file does not exist.

        Args:
            passages (List[str]): list of `q` or `r`
            indexes (List[int]): list of `id`
            prefix (str): prefix (`q` or `r`) of the generated files.
        """
        corpus_path = os.path.join(self.qgen_dir, f"{prefix}corpus.jsonl")
        if not os.path.exists(corpus_path):
            write_qgen_corpus(indexes, passages, corpus_path)
        else:
            logging.info(
                f"{corpus_path} exists. Load corpus from the existing file.")
        # self.qgen_corpus[prefix] = GenericDataLoader(
        #     data_folder=self.qgen_dir, corpus_file=f"{prefix}corpus.jsonl").load_corpus()
        return GenericDataLoader(data_folder=self.qgen_dir,
                          corpus_file=f"{prefix}corpus.jsonl").load_corpus()

    def generate_questions(self, passages, indexes, prefix, batch_sz=10):
        """Generate questions for the target passages.
        Write questions to `qgen_dir/{prefix}qgen-queries.jsonl` if the file does not exist.

        Args:
            passages (List[str]): list of `q` or `r`
            indexes (List[int]): list of `id`
            prefix (str): prefix (`q` or `r`) of the generated files.
            batch_sz (int): batch size (10 takes 16G+)
        """
        corpus = self._load_qgen_corpus(passages, indexes, prefix)
        query_path = os.path.join(self.qgen_dir, f"{prefix}qgen-queries.jsonl")
        if not os.path.exists(query_path):
            self.generator.generate(
                corpus,
                output_dir=self.qgen_dir,
                ques_per_passage=self.ques_per_passage,
                prefix=f"{prefix}qgen",
                batch_size=batch_sz
            )
            logging.info(f"Write generated queries to {query_path}.")
        else:
            logging.info(f"{query_path} exists. Use the catched queries.")

    def load_questions(self, passages, indexes, prefix):
        """Load questions generated by question generator (`self.generate_questions`).
            - output/qgen/{prefix}qgen-queries.jsonl, the format:
                {
                    "_id": "genQ1",
                    "text": "can things go both ways",
                    "metadata": {}
                }
            - output/qgen/{prefix}qgen-qrels/train.tsv
                query-id        corpus-id       score
                  genQ1             8             1
        Args:
            prefix (str): `q` or `r`.
        """
        path = os.path.join(self.qgen_dir, f"{prefix}qgen-queries.jsonl")
        if not os.path.exists(path):
            self.generate_questions(passages, indexes, prefix)

        # load generated questions
        with open(path, 'r') as f:
            lines = f.readlines()
        data = collections.defaultdict(list)
        for line in lines:
            sample = json.loads(line)
            data["query-id"].append(sample["_id"])
            data["text"].append(sample["text"])
        df = pd.DataFrame(data)

        # the mapping of question id & corpus id
        qrels_path = os.path.join(self.qgen_dir, f"{prefix}qgen-qrels/train.tsv")
        df_rel = pd.read_csv(qrels_path, sep="\t")
        df_merged = df_rel.merge(df, on=["query-id"])
        return df_merged

    def predict(self, queries, responses, q_indexes, r_indexes, indexes, exact_search_topk=3):
        num_instances = len(q_indexes)
        q_outs, r_outs = list(), list()

        # hard code: merge queries to passages for qgen
        q_passages = [" ".join(queries[q_indexes[i][0]:q_indexes[i][1]])
                      for i in range(num_instances)]
        r_passages = [" ".join(responses[r_indexes[i][0]:r_indexes[i][1]])
                      for i in range(num_instances)]
        df_q = self.load_questions(q_passages, indexes, "q")
        df_r = self.load_questions(r_passages, indexes, "r")

        for i in tqdm(range(num_instances)):
            qs, qe = q_indexes[i]
            pred_res = self.predict_one(
                queries[qs:qe], df_q[df_q["corpus-id"] == indexes[i]])
            q_outs.append(pred_res)

            rs, re = r_indexes[i]
            pred_res = self.predict_one(
                sentences=responses[rs:re],
                df_question=df_r[df_r["corpus-id"] == indexes[i]],
                top_k=exact_search_topk)
            r_outs.append(pred_res)
        return q_outs, r_outs

    def predict_one(self, sentences, df_question, top_k=3):
        corpus = self.combine_sentences_by_sliding_window(sentences)
        questions = {
            row["query-id"]: row["text"]
            for _, row in df_question.iterrows()
        }
        pred_res = self.select_sentences_by_retrieve_results(
            corpus, questions, top_k=top_k)
        # make a guess: return the first sentence of the queries
        # in case the generated questions is not good
        if pred_res is None:
            pred_res = " ".join(sentences) # ""
        return pred_res

    def combine_sentences_by_sliding_window(self, sentences, window_szs=None):
        if window_szs is None:
            window_szs = [1, 2, 3, 4]

        corpus = dict()
        corpus_id = 0
        for i in range(len(sentences)):
            for window_sz in window_szs:
                if i + window_sz >= len(sentences):
                    break
                corpus[corpus_id] = {"text": " ".join(sentences[i:i+window_sz])}
                corpus_id += 1
        return corpus

    def select_sentences_by_retrieve_results(self, corpus, questions, top_k=3):
        retrieval_results = self.dense_retriever.search(
            corpus, questions, top_k=top_k, score_function="cos_sim")

        score_dict = collections.defaultdict(list)
        for _, scores in retrieval_results.items():
            for sid, s in scores.items():
                score_dict[sid].append(s)

        if len(score_dict) > 0:
            max_idx = max(score_dict, key=lambda x: sum(
                score_dict[x]) / len(score_dict[x]))
            return corpus[max_idx]["text"]
        return corpus.get(0)

    def rerank():
        pass
