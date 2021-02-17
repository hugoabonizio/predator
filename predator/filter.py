import os
import collections
from sklearn import metrics
import datasets
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
from transformers.data.data_collator import DataCollatorWithPadding


class Filter:
    def __init__(
        self,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        device="cpu",
        model_name_or_path="distilbert-base-uncased",
    ):
        num_labels = len(set(train_labels))
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        self.config.max_length = self.tokenizer.max_model_input_sizes[
            "bert-base-cased"
            if "bert-base-cased" in self.tokenizer.max_model_input_sizes
            else "distilbert-base-uncased"
        ]
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        ).to(self.device)

        class_counts = list(collections.Counter(train_labels).values())
        total_samples = sum(class_counts)
        class_weights = [
            total_samples / class_counts[i] for i in range(len(class_counts))
        ]
        weights = [class_weights[train_labels[i]] for i in range(total_samples)]
        self.train_sampler = torch.utils.data.WeightedRandomSampler(
            torch.DoubleTensor(weights), total_samples
        )

        train_data = self.tokenizer(
            train_texts,
            padding="longest",
            max_length=self.config.max_length,
            truncation=True,
        )
        train_data.update({"label": train_labels})
        self.train_dataset = datasets.Dataset.from_dict(train_data)

        val_data = self.tokenizer(
            val_texts,
            padding="longest",
            max_length=self.config.max_length,
            truncation=True,
        )
        val_data.update({"label": val_labels})
        self.val_dataset = datasets.Dataset.from_dict(val_data)

        self.threshold_min = None
        self.threshold_max = None

    def train(self, epochs=3, batch_size=32, lr=3e-5):
        training_args = TrainingArguments(
            output_dir="./output-cls",
            no_cuda=(self.device != torch.device("cuda")),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_steps=len(self.train_dataset),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            fp16=(self.device == torch.device("cuda")),
        )

        self.trainer = _FilterTrainer(
            sampler=self.train_sampler,
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_metrics_fn,
        )
        self.trainer.train()
        return self.trainer

    def evaluate(self):
        return {
            **self.trainer.evaluate(),
            **self.trainer.evaluate(self.train_dataset, metric_key_prefix="train"),
        }

    def select(self, texts):
        if len(texts) == 0: return []

        if self.threshold_min is None or self.threshold_max is None:
            self._compute_thresholds()

        with torch.no_grad():
            tokenizer_output = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                max_length=self.config.max_length,
            )

            input_ids = tokenizer_output["input_ids"].to(self.device)
            attention_mask = tokenizer_output["attention_mask"].to(self.device)
            preds = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).logits
            output = F.softmax(preds, dim=1)

            max_value, max_index = torch.max(output, dim=1)

            threshold_mask = (max_value > self.threshold_min) & (
                max_value < self.threshold_max
            )
            filtered_texts = [
                text for (text, mask) in zip(texts, threshold_mask) if mask.item()
            ]
            filtered_labels = max_index[threshold_mask].tolist()

            return [
                [text, int(label)]
                for (text, label) in zip(filtered_texts, filtered_labels)
                if len(text) > 3
            ]

    def save(self, path):
        os.makedirs(f"{path}/model-cls", exist_ok=True)
        self.trainer.save_model(f"{path}/model-cls")
        self.tokenizer.save_pretrained(f"{path}/model-cls")

    def _compute_metrics_fn(self, pred: EvalPrediction):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        acc = metrics.accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def _compute_thresholds(self):
        confidences = []
        with torch.no_grad():
            loader = data.DataLoader(
                self.val_dataset,
                batch_size=64,
                collate_fn=DataCollatorWithPadding(self.tokenizer),
            )
            for batch in loader:
                labels = batch.pop("labels")

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                preds = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                ).logits
                output = F.softmax(preds, dim=1).cpu()
                max_value, max_index = torch.max(output, dim=1)
                confidences += max_value[max_index == labels].tolist()

        self.threshold_min = 0.7  # torch.tensor(confidences).mean().item()
        self.threshold_max = 1.0  # torch.tensor(confidences).max().item()



class _FilterTrainer(Trainer):
    def __init__(self, sampler, **kwargs):
        super().__init__(**kwargs)
        self.sampler = sampler

    def get_train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self.sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )
