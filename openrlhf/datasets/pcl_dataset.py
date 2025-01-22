from typing import Optional
import json

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .utils import zero_pad_sequences


class PCLDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        split: str = "train",
        train_file: Optional[str] = None,
        step_level: bool = False,
        period_as_delimiter: bool = False,
        padding_side: str = "right",
        rew_mul: float = 2,
        rew_add: float = -1,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.step_level = step_level
        self.period_as_delimiter = period_as_delimiter
        self.padding_side = padding_side
        self.rew_mul = rew_mul
        self.rew_add = rew_add
        # TODO: filter data samples

        # TODO: load real data
        # self.data = [("Hello: ", "world!"), ("What is 1+1?\n", "The answer is: 2")]
        self.load_data(split, train_file)

    def load_data(self, split: str, train_file: Optional[str] = None):
        if split == "train":
            filepath = "/mnt/shared/annotated/train-n5-t1.jsonl" if train_file is None else train_file
        else:
            filepath = "/mnt/shared/annotated/test.jsonl" if train_file is None else train_file
        self.data = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = self.data[index]["prompt"]
        response: str = self.data[index]["response"]
        period_as_delimiter = False
        reward = self.data[index]["reward"]
        if self.step_level:
            if (
                self.period_as_delimiter
                and prompt.find("Please reason step by step, and put your final answer within \\boxed{}.") != -1
            ):
                period_as_delimiter = True
                lines = []
                delimiters = ("\n", ". ", ".$ ")
                j = 0
                for i in range(len(response)):
                    if i == len(response) - 1 or response[j : i + 1].endswith(delimiters):
                        lines.append(response[j : i + 1])
                        j = i + 1
            else:
                lines = response.split("\n")
                response = "\n".join(lines)

        input_token = self.tokenizer(
            prompt + response + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )  # TODO
        ids = input_token["input_ids"]
        attention_mask = input_token["attention_mask"]

        state_mask = torch.zeros_like(ids)
        action_mask = torch.zeros_like(ids)
        if not self.step_level:
            # TODO: this are masks for token-wise PCL
            idx = input_token.char_to_token(len(prompt))  # first token pos of response
            state_mask[0][idx - 1 : -1] = 1
            action_mask[0][idx:] = 1
        else:
            offset = len(prompt)
            for i, line in enumerate(lines):
                idx = input_token.char_to_token(offset)
                if idx is None:
                    break
                state_mask[0][idx - 1] = 1
                if period_as_delimiter:
                    offset += len(line)
                else:
                    offset += len(line) + 1
                if i == len(lines) - 1:
                    action_mask[0][idx:] = 1
                else:
                    next_idx = input_token.char_to_token(offset)
                    action_mask[0][idx:next_idx] = 1

        return (
            ids,
            attention_mask,
            state_mask,
            action_mask,
            torch.tensor([reward * self.rew_mul + self.rew_add], dtype=torch.float, device=ids.device),
        )

    def collate_fn(self, item_list):
        ids, masks, state_masks, action_masks, rewards = zip(*item_list)
        ids = zero_pad_sequences(ids, self.padding_side, self.tokenizer.pad_token_id)
        masks = zero_pad_sequences(masks, self.padding_side)
        state_masks = zero_pad_sequences(state_masks, self.padding_side)
        action_masks = zero_pad_sequences(action_masks, self.padding_side)
        rewards = torch.stack(rewards, dim=0)
        return ids, masks, state_masks, action_masks, rewards


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/mnt/minicpm")
    dataset = PCLDataset(tokenizer, 2048)
    ids, masks, state_masks, action_masks, rewards = dataset[0]
    print(dataset[0])
    idx = torch.nonzero(action_masks)[0, 1].item()
    print(tokenizer.decode(ids[0, :idx]))
