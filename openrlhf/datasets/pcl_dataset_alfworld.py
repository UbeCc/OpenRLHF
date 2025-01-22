import json

import torch
from transformers import PreTrainedTokenizer

from .pcl_dataset import PCLDataset


class PCLDatasetAlfWorld(PCLDataset):
    def load_data(self, split: str, train_file: str | None = None):
        if split == "train":
            assert train_file is not None
            filepath = train_file
        else:
            # TODO:
            filepath = "/mnt/shared/annotated/train-alfworld.json"
        with open(filepath, "r") as f:
            self.data = json.load(f)

    def apply_chat_template(self, conversations) -> str:
        prompt = self.tokenizer.apply_chat_template(conversations, tokenize=False)
        if prompt.startswith(self.tokenizer.bos_token):
            prompt = prompt[len(self.tokenizer.bos_token) :].lstrip()
        return prompt

    def __getitem__(self, index):
        conversations = self.data[index]["conversations"]
        input_token = self.tokenizer(
            self.apply_chat_template(conversations),
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        ids = input_token["input_ids"]
        attention_mask = input_token["attention_mask"]
        state_mask = torch.zeros_like(ids)
        action_mask = torch.zeros_like(ids)

        # TODO: is this the best way?
        for i in range(3, len(conversations)):
            if conversations[i]["role"] == "assistant":
                prompt = self.apply_chat_template(conversations[:i]).rstrip()
                idx = input_token.char_to_token(len(prompt))
                if idx is None:
                    break
                next_prompt = self.apply_chat_template(conversations[: i + 1]).rstrip()
                assert next_prompt.startswith(prompt)
                next_idx = input_token.char_to_token(len(next_prompt))
                if self.step_level:
                    state_mask[0, idx - 1] = 1
                else:
                    if next_idx is None:
                        state_mask[0, idx - 1 : -1] = 1
                    else:
                        state_mask[0, idx - 1 : next_idx - 1] = 1
                action_mask[0, idx:next_idx] = 1

        return (
            ids,
            attention_mask,
            state_mask,
            action_mask,
            torch.tensor([self.data[index]["reward"]], dtype=torch.float, device=ids.device),
        )
