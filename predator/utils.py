import torch
from torch.utils import data


class LineByLineTextDataset(data.Dataset):
    def __init__(self, tokenizer, examples, max_length):
        batch_encoding = tokenizer(
            examples, add_special_tokens=True, truncation=True, max_length=max_length
        )
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class BlockTextDataset(data.Dataset):
    def __init__(self, tokenizer, examples, block_size=64):
        text = f" {tokenizer.eos_token} ".join(examples)
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        self.examples = []

        for i in range(
            0, len(tokenized_text) - block_size + 1, block_size
        ):  # Truncate in block of block_size
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i : i + block_size]
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
