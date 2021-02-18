import os
import torch
import numpy as np
import pandas as pd
import texthero as th
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from .utils import LineByLineTextDataset, BlockTextDataset


class Generator:
    def __init__(
        self, texts, labels, val_texts, device="cpu", model_name_or_path="distilgpt2"
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        max_length = max(len(i) for i in self.tokenizer(texts)["input_ids"])
        self.avg_length = int(
            np.mean([len(i) for i in self.tokenizer(texts)["input_ids"]])
        )
        self.std_length = int(
            np.std([len(i) for i in self.tokenizer(texts)["input_ids"]])
        )
        print("avg_length", self.avg_length, "std_length", self.std_length)

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.max_length = min(
            self.tokenizer.max_model_input_sizes["gpt2"],
            # (self.avg_length + 3 * self.std_length),
            float("inf"),
        )

        max_length = self.config.max_length

        print("max_length", max_length, self.config.max_length)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, config=self.config
        ).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

        df_train = pd.DataFrame({"text": texts, "label": labels})
        max_count_label = df_train["label"].value_counts().max()
        dfs_tmp = [df_train]
        for label, group in df_train.groupby("label"):
            dfs_tmp.append(group.sample(max_count_label - len(group), replace=True))
        df_balanced = pd.concat(dfs_tmp)

        self.train_dataset = BlockTextDataset(self.tokenizer, df_balanced["text"])
        self.val_dataset = BlockTextDataset(self.tokenizer, val_texts)

    def train(self, epochs, lr=5e-5, batch_size=1):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        training_args = TrainingArguments(
            output_dir="./output-lm",
            no_cuda=(self.device != torch.device("cuda")),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            # save_steps=10,
            save_total_limit=1,
            learning_rate=lr,
            evaluation_strategy="epoch",
            logging_steps=float("inf"),
            prediction_loss_only=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        self.trainer.train()
        return self.trainer

    def evaluate(self):
        return {
            **self.trainer.evaluate(),
            **self.trainer.evaluate(self.train_dataset, metric_key_prefix="train"),
        }

    def generate(
        self,
        input_text,
        top_k=40,
        top_p=None,
        temperature=1.0,
        repetition_penalty=1.0,
        num_return_sequences=4,
    ):
        max_length_input = (
            self.config.max_length
            if self.avg_length > self.config.max_length
            else self.config.max_length - self.avg_length
        )
        input_ids = self.tokenizer.encode(
            input_text,
            max_length=max_length_input,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        min_length = input_ids.shape[1] + 6
        max_length_output = min(
            input_ids.shape[1] + (self.avg_length), self.config.max_length
        )

        output = self.model.generate(
            input_ids=input_ids,
            min_length=min_length,
            max_length=max_length_output,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
        results = self.tokenizer.batch_decode(
            output[:, input_ids.size(1) :], skip_special_tokens=True
        )
        return [
            txt
            for txt in th.remove_whitespace(pd.Series(results)).tolist()
            if len(txt) > 3
        ]

    def save(self, path):
        os.makedirs(f"{path}/model-lm", exist_ok=True)
        self.trainer.save_model(f"{path}/model-lm")
        self.tokenizer.save_pretrained(f"{path}/model-lm")
