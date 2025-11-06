"""
Task-based fine-tuning utility for Hugging Face Transformers models.
Reads settings from a YAML configuration file, dynamically dispatches to the appropriate finetuner class.
Generates output directories named by model and task automatically.
"""

import os
import yaml
import argparse

import logging
import statistics
import json
import re
import csv
from typing import Dict, List, Tuple, Type
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import eval4ner.muc as muc

import evaluate
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

def write_run_json(config: Dict, output_dir: str, filename: str = "config_run.json"):
    """
    Dump the full config dict to a JSON file inside the run's output directory.
    """
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def update_master_csv(config: Dict, base_out: str, filename: str = "all_runs.csv"):
    """
    Append a row summarizing this run's config to a master CSV under base_out.
    Columns are 'run_dir' plus each config key.
    """
    master_path = os.path.join(base_out, filename)
    os.makedirs(base_out, exist_ok=True)
    # build fieldnames from config keys
    fieldnames = ["run_dir"] + list(config.keys())
    file_exists = os.path.isfile(master_path)
    with open(master_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # build a flat row, JSON-encode complex values
        run_dir_name = os.path.basename(config.get("output_dir_name", ""))
        row = {"run_dir": run_dir_name}
        for k, v in config.items():
            row[k] = json.dumps(v) if not isinstance(v, str) else v
        writer.writerow(row)


class BaseFinetuner:
    """
    Base class for task-specific finetuners.
    Handles common setup and automatic output_dir generation.
    """

    def __init__(self, config: Dict, seed: int = 42):
        self.config = config
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Seed for reproducibility
        seed_value = config.get("seed", 42)
        if isinstance(seed_value, list):
            # Use first seed for global initialization; per‑run seeds handled in subclass train()
            init_seed = int(seed_value[0]) if seed_value else 42
        else:
            init_seed = int(seed_value)
        set_seed(init_seed)
        self.logger.info(f"Random seed set to {seed}")

        # Build output directory
        model_id = config["model_name_or_path"].rsplit("/", 1)[-1]
        task_name = config.get("task_name", config.get("task", "task"))
        base_out = config.get("output_dir", "./outputs")
        run_id = config.get("run_id", "")
        out_name = f"{task_name}_{model_id}"
        if run_id:
            out_name += f"_{run_id}"
        self.output_dir = os.path.join(base_out, out_name)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self) -> DatasetDict:
        raise NotImplementedError("Subclasses must implement load_data()")

    def train(self):
        raise NotImplementedError("train() must be implemented by subclasses")


class ClassifierFinetuner(BaseFinetuner):
    """
    Finetuner for sequence classification tasks.
    """

    def load_data(self) -> DatasetDict:
        cfg = self.config
        train_path = cfg.get("train_path")
        test_path = cfg.get("test_path")
        if not train_path or not test_path:
            raise ValueError("`train_path` and `test_path` must be specified in config")
        val_path = cfg.get("validation_path", test_path)

        splits = {}
        for name, path in (
            ("train", train_path),
            ("validation", val_path),
            ("test", test_path),
        ):
            ext = os.path.splitext(path)[1].lstrip(".").lower()
            fmt = ext if ext in ("json", "csv") else cfg.get("file_format")
            if fmt not in ("json", "csv"):
                raise ValueError(f"Unsupported format `{fmt}` for {name} data")
            self.logger.info(f"Loading {fmt.upper()} {name} data from {path}")
            ds = load_dataset(fmt, data_files={name: path})[name]
            splits[name] = ds

        ds = DatasetDict(splits)
        label_col = cfg.get("label_column")
        if label_col:
            for split in ds.values():
                if label_col not in split.column_names:
                    raise ValueError(f"Label column {label_col} missing in {split}")
            labels = sorted(set(ds["train"][label_col]))
            self.label2id = {lbl: i for i, lbl in enumerate(labels)}
            ds = ds.map(
                lambda ex: {"labels": self.label2id[ex[label_col]]}, batched=False
            )
            ds = ds.remove_columns(label_col)

        return ds

    def _tokenize(self, ds: DatasetDict) -> DatasetDict:
        cfg = self.config
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name_or_path"], use_fast=True
        )
        text1 = cfg["text_column"]
        text2 = cfg.get("text_pair_column")
        max_len = cfg.get("max_length", 256)

        def preprocess(batch):
            # ensure inputs are lists of strings
            texts = batch[text1]
            if not isinstance(texts, list):
                texts = [texts]
            clean_texts = [t if isinstance(t, str) else "" for t in texts]

            if text2:
                pairs = batch[text2]
                if not isinstance(pairs, list):
                    pairs = [pairs]
                clean_pairs = [p if isinstance(p, str) else "" for p in pairs]
                enc = tokenizer(
                    clean_texts,
                    clean_pairs,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                )
            else:
                enc = tokenizer(
                    clean_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                )

            if "labels" in batch:
                enc["labels"] = batch["labels"]
            return enc

        remove_cols = [text1] + ([text2] if text2 else [])
        return ds.map(preprocess, batched=True, remove_columns=remove_cols)

    def build_model_and_metric(self):
        cfg = self.config
        num_labels = cfg.get("num_labels", len(getattr(self, "label2id", {})))
        self.logger.info(f"Configuring model with {num_labels} labels")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name_or_path"], use_fast=True
        )
        model_config = AutoConfig.from_pretrained(
            cfg["model_name_or_path"], num_labels=num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name_or_path"], config=model_config
        )

        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")
        self.logger.info("Metrics loaded: accuracy, f1, precision, recall")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": self.accuracy.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "f1": self.f1.compute(
                predictions=preds, references=labels, average="micro"
            )["f1"],
            "precision": self.precision.compute(
                predictions=preds, references=labels, average="micro"
            )["precision"],
            "recall": self.recall.compute(
                predictions=preds, references=labels, average="micro"
            )["recall"],
        }

    def train(self):
        cfg = self.config
        seeds = cfg.get("seed", 42)
        if not isinstance(seeds, list):
            seeds = [seeds]

        raw = self.load_data()
        tokenized_ds = self._tokenize(raw)

        metrics_path = f"{self.output_dir}.metric.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        for seed in seeds:
            self.build_model_and_metric()
            args = TrainingArguments(
                output_dir=self.output_dir,
                seed=int(seed),
                evaluation_strategy=cfg.get("save_strategy", "epoch"),
                save_strategy=cfg.get("save_strategy", "epoch"),
                eval_steps=int(cfg.get("eval_steps", 200)),
                save_steps=int(cfg.get("save_steps", 200)),
                logging_dir=os.path.join(self.output_dir, "logs"),
                logging_steps=int(cfg.get("logging_steps", 1000)),
                learning_rate=float(cfg.get("learning_rate", 5e-5)),
                per_device_train_batch_size=int(cfg.get("train_batch_size", 16)),
                per_device_eval_batch_size=int(cfg.get("eval_batch_size", 16)),
                weight_decay=float(cfg.get("weight_decay", 0.0)),
                num_train_epochs=int(cfg.get("num_train_epochs", 3)),
                load_best_model_at_end=bool(cfg.get("load_best_model_at_end", False)),
                metric_for_best_model=cfg.get("metric_for_best_model", "accuracy"),
                greater_is_better=bool(cfg.get("greater_is_better", True)),
                report_to=cfg.get("report_to", []),
                fp16=bool(cfg.get("fp16", False)),
            )
            trainer = Trainer(
                model=self.model,
                args=args,
                train_dataset=tokenized_ds["train"],
                eval_dataset=tokenized_ds.get("validation"),
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )
            self.logger.info(f"Starting classification training with seed {seed}...")
            trainer.train()
            self.logger.info("Training complete.")

            results = trainer.evaluate()
            with open(metrics_path, "a", encoding="utf-8") as mf:
                json.dump(results, mf, indent=2)
                mf.write("\n")
            self.logger.info(
                f"Appended evaluation metrics for seed {seed} to {metrics_path}"
            )

        # Compute average and standard deviation of eval_f1 across seeds
        with open(metrics_path, "r", encoding="utf-8") as mf:
            buffer = ""
            f1_vals = []
            for line in mf:
                buffer += line
                if line.strip().endswith("}"):
                    try:
                        entry = json.loads(buffer)
                        f1_vals.append(entry["eval_f1"])
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping malformed JSON entry: {e}")
                    buffer = ""

        if f1_vals:
            avg_f1 = sum(f1_vals) / len(f1_vals)
            self.logger.info(f"Average eval_f1 across seeds: {avg_f1:.4f}")
            with open(metrics_path, "a", encoding="utf-8") as mf:
                json.dump({"average_eval_f1": round(avg_f1, 4)}, mf)
                mf.write("\n")
            std_f1 = statistics.stdev(f1_vals) if len(f1_vals) > 1 else 0.0
            self.logger.info(
                f"Standard deviation of eval_f1 across seeds: {std_f1:.4f}"
            )
            with open(metrics_path, "a", encoding="utf-8") as mf:
                json.dump({"std_eval_f1": round(std_f1, 4)}, mf)
                mf.write("\n")

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.logger.info(f"Model and tokenizer saved to {self.output_dir}")


class ExtractionFinetuner(BaseFinetuner):
    """
    Finetuner for token‑classification extraction tasks.
    """

    task_name: str = "extraction"

    def load_data(self) -> DatasetDict:
        cfg = self.config
        train_path = cfg["train_path"]
        test_path = cfg["test_path"]
        val_path = cfg.get("validation_path", test_path)

        splits = {}
        for name, path in (
            ("train", train_path),
            ("validation", val_path),
            ("test", test_path),
        ):
            ext = os.path.splitext(path)[1].lstrip(".").lower()
            fmt = ext if ext in ("json", "csv") else cfg.get("file_format")
            if fmt not in ("json", "csv"):
                raise ValueError(f"Unsupported format `{fmt}` for {name}")
            self.logger.info(f"Loading {fmt.upper()} {name} split from {path}")
            splits[name] = load_dataset(fmt, data_files={name: path})[name]

        ds = DatasetDict(splits)

        # Build label ↔ id mapping from all labels found in TRAIN split
        # (we always add an explicit "O" label for outside / background)
        lbl_col = cfg["label_column"]
        labels_in_data = {
            lbl
            for sample in ds["train"][lbl_col] or []  # safety: None → []
            for lbl, _ in sample
        }
        labels_in_data.add("O")
        self.label2id: Dict[str, int] = {
            l: i for i, l in enumerate(sorted(labels_in_data))
        }
        self.id2label: Dict[int, str] = {i: l for l, i in self.label2id.items()}

        # Keep for later
        self.text_col = cfg["text_column"]
        self.text_pair_col = cfg.get("text_pair_column")
        self.span_col = lbl_col
        return ds

    def _tokenize(self, ds: DatasetDict) -> DatasetDict:
        cfg = self.config
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name_or_path"], use_fast=True
        )

        max_len = cfg.get("max_length", 512)
        label2id = self.label2id  # local shortcut
        span_col = self.span_col
        text1 = self.text_col
        text2 = self.text_pair_col

        def encode(example):
            if text2:
                full_txt = f"{example[text1]} {example[text2]}"
                enc = tokenizer(
                    example[text1],
                    example[text2],
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_offsets_mapping=True,
                )
            else:
                full_txt = example[text1]
                enc = tokenizer(
                    full_txt,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_offsets_mapping=True,
                )

            offsets = enc.pop("offset_mapping")  # we only need it here
            # build per‑token label ids
            token_labels = [label2id["O"]] * len(offsets)
            for lbl, span_text in example[span_col]:
                # match *all* occurrences of the span text in the full string
                for m in re.finditer(re.escape(span_text), full_txt):
                    s, e = m.span()
                    for tidx, (tok_s, tok_e) in enumerate(offsets):
                        if tok_s >= s and tok_e <= e:
                            token_labels[tidx] = label2id[lbl]

            enc["labels"] = token_labels  # for Trainer
            enc[span_col] = example[span_col]  # untouched GT spans
            return enc

        remove_cols = [text1] + ([text2] if text2 else [])
        return ds.map(encode, remove_columns=remove_cols)

    #  Model / metric initialisation helpers

    def build_model_and_metric(self):
        cfg = self.config
        num_labels = len(self.label2id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name_or_path"], use_fast=True
        )
        model_cfg = AutoConfig.from_pretrained(
            cfg["model_name_or_path"],
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            cfg["model_name_or_path"], config=model_cfg
        )
        self.logger.info(f"Model ready with {num_labels} labels")

    # Helper: convert per‑token label ids back to contiguous spans
    def _ids_to_spans(
        self, label_ids: List[int], input_ids: List[int]
    ) -> List[Tuple[str, str]]:
        spans, cur_lbl, cur_toks = [], None, []
        for lbl_id, tok_id in zip(label_ids, input_ids):
            lbl = self.id2label[lbl_id]
            tok = self.tokenizer.convert_ids_to_tokens([tok_id])[0]

            if tok in self.tokenizer.all_special_tokens:
                continue  # skip [CLS], [SEP], etc.

            if lbl != "O":
                if lbl == cur_lbl:
                    cur_toks.append(tok)
                else:
                    if cur_lbl:  # flush previous
                        spans.append(
                            (
                                cur_lbl,
                                self.tokenizer.convert_tokens_to_string(
                                    cur_toks
                                ).strip(),
                            )
                        )
                    cur_lbl, cur_toks = lbl, [tok]
            else:  # lbl == "O"
                if cur_lbl:
                    spans.append(
                        (
                            cur_lbl,
                            self.tokenizer.convert_tokens_to_string(cur_toks).strip(),
                        )
                    )
                cur_lbl, cur_toks = None, []

        if cur_lbl:  # flush tail
            spans.append(
                (cur_lbl, self.tokenizer.convert_tokens_to_string(cur_toks).strip())
            )
        return spans

    def train(self):
        cfg = self.config
        seeds = cfg.get("seed", 42)
        if not isinstance(seeds, list):
            seeds = [seeds]

        # collect MUC-F1 across seeds
        f1_vals_extr: List[float] = []

        raw_ds = self.load_data()
        tokenised_ds = self._tokenize(raw_ds)

        metrics_path = f"{self.output_dir}.metric.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        for seed in seeds:
            # model / tokenizer fresh per seed
            self.build_model_and_metric()

            args = TrainingArguments(
                output_dir=self.output_dir,
                seed=int(seed),
                eval_strategy=cfg.get("save_strategy", "epoch"),
                save_strategy=cfg.get("save_strategy", "epoch"),
                save_total_limit=int(cfg.get("save_total_limit", 3)),
                eval_steps=int(cfg.get("eval_steps", 500)),
                save_steps=int(cfg.get("save_steps", 500)),
                logging_dir=os.path.join(self.output_dir, "logs"),
                logging_steps=int(cfg.get("logging_steps", 1000)),
                learning_rate=float(cfg.get("learning_rate", 5e-5)),
                per_device_train_batch_size=int(cfg.get("train_batch_size", 8)),
                per_device_eval_batch_size=int(cfg.get("eval_batch_size", 8)),
                weight_decay=float(cfg.get("weight_decay", 0.0)),
                num_train_epochs=int(cfg.get("num_train_epochs", 1)),
                report_to=cfg.get("report_to", []),
                fp16=bool(cfg.get("fp16", False)),
            )

            trainer = Trainer(
                model=self.model,
                args=args,
                train_dataset=tokenised_ds["train"],
                eval_dataset=tokenised_ds.get("validation"),
                data_collator=DataCollatorForTokenClassification(self.tokenizer),
            )

            self.logger.info(f"Starting extraction training with seed {seed}")
            trainer.train()
            self.logger.info("Training complete")

            # Span‑level evaluation with eval4ner
            preds = trainer.predict(tokenised_ds["test"])
            logits = preds.predictions  # (N, L, C)
            true_spans = tokenised_ds["test"][self.span_col]

            # build raw full‑text strings for MUC
            if self.text_pair_col:
                texts = [
                    a + b
                    for a, b in zip(
                        raw_ds["test"][self.text_col],
                        raw_ds["test"][self.text_pair_col],
                    )
                ]
            else:
                texts = raw_ds["test"][self.text_col]

            pred_spans_all = []
            for idx, pred_logits in enumerate(logits):
                label_ids = np.argmax(pred_logits, axis=-1).tolist()
                input_ids = tokenised_ds["test"]["input_ids"][idx]
                pred_spans_all.append(self._ids_to_spans(label_ids, input_ids))

            # Normalize and dedupe spans for predictions and ground truth
            def normalize(spans):
                out = []
                for s in spans:
                    if isinstance(s, list) and len(s) == 2:
                        out.append((s[0], s[1]))
                    elif isinstance(s, tuple) and len(s) == 2:
                        out.append(s)
                return out

            def dedupe(spans):
                seen = set()
                out = []
                for s in spans:
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
                return out

            pred_spans_clean = [dedupe(normalize(p)) for p in pred_spans_all]
            gt_spans_clean = [dedupe(normalize(g)) for g in true_spans]

            muc_results = muc.evaluate_all(
                pred_spans_clean, gt_spans_clean, texts, False
            )

            # append metrics
            with open(metrics_path, "a", encoding="utf-8") as mf:
                json.dump(muc_results, mf, indent=2)
                mf.write("\n")
            self.logger.info(f"Appended MUC metrics for seed {seed} to {metrics_path}")
            f1_vals_extr.append(muc_results["f1"])
            # persist artefacts
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model + tokenizer saved to {self.output_dir}")
            # after all seeds, compute standard deviation of MUC F1
        if f1_vals_extr:
            std_f1_extr = (
                statistics.stdev(f1_vals_extr) if len(f1_vals_extr) > 1 else 0.0
            )
            self.logger.info(
                f"Standard deviation of MUC F1 across seeds: {std_f1_extr:.4f}"
            )
            with open(metrics_path, "a", encoding="utf-8") as mf:
                json.dump({"std_muc_f1": round(std_f1_extr, 4)}, mf)
                mf.write("\n")


# Registry maps task names to Finetuner classes
FINETUNER_REGISTRY: Dict[str, Type[BaseFinetuner]] = {
    "classification": ClassifierFinetuner,
    "extraction": ExtractionFinetuner,
    # extend with other tasks: 'seq2seq': Seq2SeqFinetuner, etc.
}


def main():
    parser = argparse.ArgumentParser(
        description="Finetuning encoders with config_models_fine_tuning.yaml"
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="config_models_fine_tuning.yaml"
    )
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_file))

    task = config.get("task", "classification")
    finetuner_cls = FINETUNER_REGISTRY.get(task)
    if not finetuner_cls:
        raise ValueError(
            f"Unknown task '{task}'. Available: {list(FINETUNER_REGISTRY)}"
        )

    finetuner = finetuner_cls(config)
    finetuner.train()
    write_run_json(config, finetuner.output_dir)
    update_master_csv(config, config.get("output_dir", "./outputs"))


if __name__ == "__main__":
    main()