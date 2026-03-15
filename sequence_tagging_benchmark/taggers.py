import time
from typing import Any, Callable
from abc import ABC, abstractmethod
import numpy as np
import psutil
import os
import gc
from sklearn.metrics import f1_score as sklearn_f1
from seqeval.metrics import f1_score as seqeval_f1
from nltk.tag import HiddenMarkovModelTagger
import sklearn_crfsuite
from torchao.core.config import AOBaseConfig
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForTokenClassification
from torchao.quantization import (
    quantize_,
)
from hmmlearn import hmm
import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification


class BaseSequenceLabellingTagger(ABC):
    def _get_current_memory_mb(self) -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @abstractmethod
    def train(self, train_tokens: list[list[str]], train_tags: list[list[str]]) -> None:
        pass

    @abstractmethod
    def predict(
        self, sentences: list[list[str]], batch_size: int = 1
    ) -> list[list[str]]:
        pass

    def evaluate(
        self,
        test_tokens: list[list[str]],
        test_tags: list[list[str]],
        batch_size: int = 1,
        warmup_iters: int = 5,
        use_smart_batching: bool = False,
        evaluate_sequence: bool = False,
    ) -> dict[str, Any]:
        # Warmup
        if len(test_tokens) > 0:
            for _ in range(warmup_iters):
                _ = self.predict([test_tokens[0]], batch_size=1)

        # Smart Batching: Sort by length to minimize padding
        if use_smart_batching:
            indices = list(range(len(test_tokens)))
            # Sort by sequence length
            sorted_data = sorted(
                zip(test_tokens, test_tags, indices), key=lambda x: len(x[0])
            )
            eval_tokens, eval_tags, original_indices = zip(*sorted_data)
        else:
            eval_tokens, eval_tags = test_tokens, test_tags
            original_indices = list(range(len(test_tokens)))

        latencies = []
        all_preds = []

        # Force garbage collection to get accurate memory usage
        gc.collect()
        mem_at_start = self._get_current_memory_mb()
        peak_inference_mem = mem_at_start

        # Batching and Inference
        for i in tqdm(range(0, len(eval_tokens), batch_size), desc="Benchmarking"):
            batch_tokens = list(eval_tokens[i : i + batch_size])

            start_t = time.perf_counter()
            batch_pred_tags = self.predict(batch_tokens, batch_size=batch_size)
            batch_latency_ms = (time.perf_counter() - start_t) * 1000

            # Distribute latency across samples
            latencies.extend([batch_latency_ms / len(batch_tokens)] * len(batch_tokens))
            all_preds.extend(batch_pred_tags)

            # Update peak inference memory
            current_mem = self._get_current_memory_mb()
            if current_mem > peak_inference_mem:
                peak_inference_mem = current_mem

        # Re-sort to original order for correct accuracy mapping
        reindexed_preds = [None] * len(all_preds)
        for i, orig_idx in enumerate(original_indices):
            reindexed_preds[orig_idx] = all_preds[i]

        y_pred_aligned_seqs = []
        y_true_aligned_seqs = []

        for pred_tags, true_tags in zip(reindexed_preds, test_tags):
            len_true = len(true_tags)
            len_pred = len(pred_tags)

            # Pad with "O" instead of "X" to ensure seqeval doesn't crash on invalid BIO tags
            if len_pred > len_true:
                aligned_pred = pred_tags[:len_true]
            elif len_pred < len_true:
                aligned_pred = pred_tags + ["O"] * (len_true - len_pred)
            else:
                aligned_pred = pred_tags

            y_pred_aligned_seqs.append(aligned_pred)
            y_true_aligned_seqs.append(true_tags)

        # Flatten just for the base accuracy score calculation
        y_true_flat = [tag for seq in y_true_aligned_seqs for tag in seq]
        y_pred_flat = [tag for seq in y_pred_aligned_seqs for tag in seq]

        # Force garbage collection to get accurate memory usage
        gc.collect()
        mem_at_end = self._get_current_memory_mb()
        peak_inference_mem = max(peak_inference_mem, mem_at_end)

        if evaluate_sequence:
            micro_f1 = seqeval_f1(
                y_true_aligned_seqs, y_pred_aligned_seqs, average="micro"
            )
            macro_f1 = seqeval_f1(
                y_true_aligned_seqs, y_pred_aligned_seqs, average="macro"
            )
        else:
            micro_f1 = sklearn_f1(
                y_true_flat, y_pred_flat, average="micro", zero_division=0
            )
            macro_f1 = sklearn_f1(
                y_true_flat, y_pred_flat, average="macro", zero_division=0
            )
        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "p50_latency_ms": np.percentile(latencies, 50),
            "p90_latency_ms": np.percentile(latencies, 90),
            "p99_latency_ms": np.percentile(latencies, 99),
            "latency_variance_ms": np.var(latencies),
            "throughput_seq_per_sec": len(test_tokens) / (np.sum(latencies) / 1000),
            "inference_memory_delta_mb": max(0, peak_inference_mem - mem_at_start),
            "final_memory_retained_mb": mem_at_end - mem_at_start,
        }


class NltkHMMTagger(BaseSequenceLabellingTagger):
    def __init__(self):
        self.model = None

    def train(self, train_tokens: list[list[str]], train_tags: list[list[str]]) -> None:
        train_data = [
            list(zip(tokens, tags)) for tokens, tags in zip(train_tokens, train_tags)
        ]
        self.model = HiddenMarkovModelTagger.train(train_data)

    def predict(
        self, sentences: list[list[str]], batch_size: int = 1
    ) -> list[list[str]]:
        batch_preds = []
        for tokens in sentences:
            # HMM tagger returns a list of (token, tag) tuples
            tagged = self.model.tag(tokens)
            batch_preds.append([tag for _, tag in tagged])
        return batch_preds


class HmmlearnTagger(BaseSequenceLabellingTagger):
    def __init__(self):
        self.model = None
        self.tag2id = {}
        self.id2tag = {}
        self.word2id = {}
        self.unk_id = 0

    def train(self, train_tokens: list[list[str]], train_tags: list[list[str]]) -> None:
        # Build Vocabularies
        unique_tags = list(set(tag for tags in train_tags for tag in tags))
        self.tag2id = {tag: i for i, tag in enumerate(unique_tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

        unique_words = list(set(word for tokens in train_tokens for word in tokens))
        self.word2id = {"<UNK>": 0}
        for word in unique_words:
            self.word2id[word] = len(self.word2id)

        n_states = len(self.tag2id)
        n_features = len(self.word2id)

        # Initialize Counts
        start_counts = np.zeros(n_states)
        trans_counts = np.zeros((n_states, n_states))
        emiss_counts = np.zeros((n_states, n_features))

        # Populate Counts from Data
        for tokens, tags in zip(train_tokens, train_tags):
            if not tags:
                continue

            start_counts[self.tag2id[tags[0]]] += 1

            for i in range(len(tags)):
                tag_id = self.tag2id[tags[i]]
                word_id = self.word2id[tokens[i]]

                emiss_counts[tag_id, word_id] += 1

                if i > 0:
                    prev_tag_id = self.tag2id[tags[i - 1]]
                    trans_counts[prev_tag_id, tag_id] += 1

        # Convert Counts to Probabilities (with Laplace smoothing)
        eps = 1e-6

        startprob = start_counts + eps
        startprob /= startprob.sum()

        transmat = trans_counts + eps
        transmat /= transmat.sum(axis=1, keepdims=True)

        emiss_counts[:, self.unk_id] = eps  # Base probability for unknown words
        emissionprob = emiss_counts + eps
        emissionprob /= emissionprob.sum(axis=1, keepdims=True)

        # Initialize hmmlearn CategoricalHMM
        self.model = hmm.CategoricalHMM(n_components=n_states, init_params="")
        self.model.n_features = n_features
        self.model.startprob_ = startprob
        self.model.transmat_ = transmat
        self.model.emissionprob_ = emissionprob

    def predict(
        self, sentences: list[list[str]], batch_size: int = 1
    ) -> list[list[str]]:
        batch_preds = []
        for tokens in sentences:
            if not tokens:
                batch_preds.append([])
                continue

            # Convert words to integer IDs, falling back to <UNK> (0)
            # hmmlearn expects shape (n_samples, 1)
            X = np.array([[self.word2id.get(w, self.unk_id)] for w in tokens])

            # hmmlearn's predict method runs the optimized Viterbi algorithm
            state_sequence = self.model.predict(X)

            batch_preds.append([self.id2tag[state_id] for state_id in state_sequence])

        return batch_preds


class CRFTagger(BaseSequenceLabellingTagger):
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=50,
            all_possible_transitions=True,
        )

    def _extract_features(self, tokens: list[str]) -> list[dict[str, Any]]:
        lowered_tokens = [t.lower() for t in tokens]

        features = []
        for i, token in enumerate(tokens):
            f = {
                "bias": 1.0,
                "word.lower()": lowered_tokens[i],
                "word[-3:]": token[-3:],
                "word.isupper()": token.isupper(),
                "word.istitle()": token.istitle(),
                "word.isdigit()": token.isdigit(),
                "BOS": i == 0,
            }

            if i > 0:
                f["-1:word.lower()"] = lowered_tokens[i - 1]
            else:
                f["-1:word.lower()"] = ""

            features.append(f)

        return features

    def train(self, train_tokens: list[list[str]], train_tags: list[list[str]]) -> None:
        X_train = [self._extract_features(tokens) for tokens in train_tokens]
        self.model.fit(X_train, train_tags)

    def predict(
        self, sentences: list[list[str]], batch_size: int = 1
    ) -> list[list[str]]:
        X_batch = [self._extract_features(tokens) for tokens in sentences]
        return self.model.predict(X_batch)


class BiLSTMTagger(BaseSequenceLabellingTagger):
    def __init__(
        self,
        model_name: str,
        tag_type: str,
        id2label: dict[int, str] | None = None,
        compile_model: bool = True,
    ):
        self.model = SequenceTagger.load(model_name)
        self.tag_type = tag_type
        self.id2label = id2label

        if compile_model:
            self.model = torch.compile(self.model, mode="max-autotune")

    def train(self, train_tokens: list[list[str]], train_tags: list[list[str]]) -> None:
        # Pre-trained model, no training required
        pass

    @torch.inference_mode()
    def predict(
        self, sentences: list[list[str]], batch_size: int = 1
    ) -> list[list[str]]:
        # Convert list of token lists into list of Flair Sentence objects
        flair_sentences = [Sentence(tokens) for tokens in sentences]

        # Flair handles batching natively
        self.model.predict(flair_sentences)

        batch_preds = []
        for sentence in flair_sentences:
            if self.tag_type == "ner":
                bio_tags = ["O"] * len(sentence.tokens)
                for span in sentence.get_spans(self.tag_type):
                    for i, token in enumerate(span.tokens):
                        list_idx = token.idx - 1
                        if i == 0:
                            bio_tags[list_idx] = f"B-{span.tag}"
                        else:
                            bio_tags[list_idx] = f"I-{span.tag}"
                batch_preds.append(bio_tags)

            else:
                batch_preds.append(
                    [token.get_label(self.tag_type).value for token in sentence.tokens]
                )
        return batch_preds


class QuantizedBiLSTMTagger(BiLSTMTagger):
    def __init__(
        self,
        model_name: str,
        tag_type: str,
        quantize_config: AOBaseConfig,
        id2label: dict[int, str] | None = None,
        compile_model: bool = True,
    ):
        # Run the parent class initialization (loads the Flair model and sets to eval mode)
        # We don't compile the model here, because we will do this for quantized model
        super().__init__(
            model_name=model_name,
            tag_type=tag_type,
            id2label=id2label,
            compile_model=False,
        )
        # Apply quantization
        quantize_(self.model, quantize_config)
        if compile_model:
            self.model = torch.compile(self.model, mode="max-autotune")


class TransformerTagger(BaseSequenceLabellingTagger):
    def __init__(
        self,
        model_name: str,
        id2label: dict[int, str] | None = None,
        compile_model: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

        if compile_model:
            self.model = torch.compile(self.model, mode="max-autotune")
        if id2label:
            self.id2label = id2label
        else:
            self.id2label = {
                i: label for i, label in enumerate(self.model.config.id2label)
            }

    def train(self, train_tokens: list[list[str]], train_tags: list[list[str]]) -> None:
        # Pre-trained model, no training required for this benchmark
        pass

    @torch.inference_mode()
    def predict(
        self, sentences: list[list[str]], batch_size: int = 1
    ) -> list[list[str]]:
        # Tokenize with padding for batching
        inputs = self.tokenizer(
            sentences,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        # Get predictions for the batch
        predictions = torch.argmax(outputs.logits, dim=-1).tolist()

        batch_aligned_preds = []

        # Align subword predictions back to original word tokens for each sentence in the batch
        for batch_idx, pred_seq in enumerate(predictions):
            word_ids = inputs.word_ids(batch_index=batch_idx)
            aligned_preds = []
            current_word = None

            for word_id, pred in zip(word_ids, pred_seq):
                if word_id is not None and word_id != current_word:
                    aligned_preds.append(self.id2label[pred])
                    current_word = word_id

            batch_aligned_preds.append(aligned_preds)

        return batch_aligned_preds


class QuantizedTransformerTagger(TransformerTagger):
    def __init__(
        self,
        model_name: str,
        quantize_config: AOBaseConfig,
        id2label: dict[int, str] | None = None,
        compile_model: bool = True,
    ):
        # Run the parent class initialization (loads model, tokenizer, labels and sets to eval mode)
        # We don't compile the model here, because we will do this for quantized model
        super().__init__(model_name=model_name, id2label=id2label, compile_model=False)
        # Apply quantization
        quantize_(self.model, quantize_config)
        if compile_model:
            self.model = torch.compile(self.model, mode="max-autotune")


class ONNXTransformerTagger(TransformerTagger):
    def __init__(
        self,
        model_name: str,
        id2label: dict[int, str] | None = None,
    ):
        super().__init__(
            model_name=model_name,
            id2label=id2label,
            compile_model=False,
        )
        # We don't need the PyTorch model anymore
        del self.model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.model = ORTModelForTokenClassification.from_pretrained(
            model_name,
            export=True,
            session_options=sess_options,
        )
