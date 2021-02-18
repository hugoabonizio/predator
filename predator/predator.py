import os
import collections
import pandas as pd
from tqdm.auto import tqdm
from .generator import Generator
from .filter import Filter


class Predator:
    def __init__(
        self,
        df_train,
        df_val,
        device,
        num_majority_classes=0,
        path=None,
        generator_kwargs={},
        filter_kwargs={},
    ):
        self.df_train = df_train.copy()
        self.df_val = df_val
        self.num_majority_classes = num_majority_classes

        if num_majority_classes > 0:
            df_train_lm = self.df_train[
                self.df_train["label"].isin(
                    self.df_train["label"].value_counts().index[1:]
                )
            ]
            df_val_lm = self.df_val[
                self.df_val["label"].isin(self.df_val["label"].value_counts().index[1:])
            ]
        else:
            df_train_lm = df_train
            df_val_lm = df_val

        if path is not None:
            if os.path.exists(f"{path}/model-lm"):
                generator_kwargs["model_name_or_path"] = f"{path}/model-lm"
            if os.path.exists(f"{path}/model-cls"):
                filter_kwargs["model_name_or_path"] = f"{path}/model-cls"

        self.generator = Generator(
            texts=df_train_lm["text"].tolist(),
            labels=df_train_lm["label"].tolist(),
            val_texts=df_val_lm["text"].tolist(),
            device=device,
            **generator_kwargs,
        )
        self.filter = Filter(
            train_texts=df_train["text"].tolist(),
            train_labels=df_train["label"].tolist(),
            val_texts=df_val["text"].tolist(),
            val_labels=df_val["label"].tolist(),
            device=device,
            **filter_kwargs,
        )

    def train(
        self,
        generator_epochs=1,
        generator_batch_size=1,
        generator_lr=5e-5,
        filter_epochs=3,
        filter_batch_size=32,
        filter_lr=3e-5,
    ):
        self.generator.train(
            epochs=generator_epochs, batch_size=generator_batch_size, lr=generator_lr
        )
        self.filter.train(
            epochs=filter_epochs, batch_size=filter_batch_size, lr=filter_lr
        )

    def evaluate(self):
        eval_filter = self.filter.evaluate()
        eval_filter.pop("epoch", None)
        eval_generator = self.generator.evaluate()
        eval_generator.pop("epoch", None)
        return {
            **{"filter_" + k: v for (k, v) in eval_filter.items()},
            **{"generator_" + k: v for (k, v) in eval_generator.items()},
        }

    def augment(
        self,
        augment_ratio=1.0,
        num_inputs=3,
        min_threshold=0.6572,
        max_threshold=0.8127,
        generator_args={},
        max_iterations=float("inf"),
    ):
        counter = collections.Counter(self.df_train["label"]).most_common()
        majority_class = counter[0][0]
        target_size = int(
            len(self.df_train[self.df_train["label"] == majority_class]) * augment_ratio
        )

        minority_class = counter[-1][0]
        minority_size = len(self.df_train[self.df_train["label"] == minority_class])

        classes_to_generate = set(self.df_train["label"].tolist())
        if augment_ratio == 1.0:
            classes_to_generate -= {majority_class}

        samples_to_create = target_size * len(classes_to_generate) - len(
            self.df_train[self.df_train["label"] != majority_class]
        )

        generated_samples = {
            c: self.df_train.query(f"label == '{c}'")["text"].tolist()
            for c in classes_to_generate
        }

        i = 0
        with tqdm(total=samples_to_create, desc="Augmentation") as pbar:
            while minority_size < target_size and i < max_iterations:
                inputs = "".join(
                    [
                        f" {t} {self.generator.tokenizer.eos_token} "
                        for t in self.df_train[
                            self.df_train["label"] == minority_class
                        ].sample(num_inputs)["text"]
                    ]
                )

                generated = self.generator.generate(inputs, **generator_args)
                selected = self.filter.select(generated)
                selected = [
                    [txt, label]
                    for (txt, label) in selected
                    if txt not in self.df_train["text"].tolist()
                ]

                if augment_ratio == 1.0:
                    selected = [
                        [txt, label]
                        for (txt, label) in selected
                        if label != majority_class
                    ]

                prev_train_size = sum(
                    [len(_samples) for (_, _samples) in generated_samples.items()]
                )

                for (txt, label) in selected:
                    if len(generated_samples[label]) < target_size:
                        generated_samples[label].append(txt)

                minority_class = min(
                    generated_samples, key=lambda x: len(generated_samples[x])
                )
                minority_size = min(
                    [len(_samples) for (_, _samples) in generated_samples.items()]
                )

                i += 1
                pbar.update(
                    sum([len(_samples) for (_, _samples) in generated_samples.items()])
                    - prev_train_size
                )

            generated_samples_list = []
            for (_class, _generated_samples) in generated_samples.items():
                for _generated_sample in _generated_samples:
                    generated_samples_list.append([_generated_sample, _class])

        self.df_train = pd.concat(
            [
                self.df_train[~self.df_train["label"].isin(classes_to_generate)],
                pd.DataFrame(
                    {
                        "text": [s[0] for s in generated_samples_list],
                        "label": [s[1] for s in generated_samples_list],
                    }
                ),
            ]
        )

        self.df_train["label"] = self.df_train["label"].astype(int)
        return self.df_train

    def save(self, path):
        self.generator.save(path)
        self.filter.save(path)
