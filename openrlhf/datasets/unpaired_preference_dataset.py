import copy
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import zero_pad_sequences, get_ranges


def preprocess_data(
    data, input_template=None, input_key=None, output_key=None, label_key=None, apply_chat_template=None,
):
    """
    Preprocess data from raw dataset to prompt, response, label

    Args:
        data: raw data from dataset
    """
    label = data[label_key]

    if apply_chat_template:
        if output_key:
            prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
            if data[output_key]:
                response = apply_chat_template(data[input_key] + data[output_key], tokenize=False)[len(prompt) :]
            else:
                response = ""
        else:
            prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=False).rstrip()
            response = None
    else:
        prompt = data[input_key]
        response = data[output_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt, response, label


class UnpairedPreferenceDataset(Dataset):
    """
    Unpaired preference dataset for algorithm, like KTO

    Args:
        dataset: raw dataset
        self.tokenizer: self.tokenizer for model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        num_processors=8,
        multiple_of=1,
        multiturn=False
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.multiturn = multiturn

        # chat_template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.label_key = getattr(self.strategy.args, "label_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.labels = processed_dataset["label"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None

    def process_data(self, data):
        if self.multiturn and self.output_key:
            if type(data[self.output_key]) == str:
                data[self.output_key] = [data[self.output_key]]
            data[self.input_key].extend(data[self.output_key])
            data[self.output_key] = None

        if self.multiturn:
            assert not self.output_key or not data[self.output_key], "You should put the whole trajactory into data[input_key] and do not set output_key"
            assert type(data[self.input_key]) == list, "You should put the whole trajactory into data[input_key] as list"
            response_ranges = get_ranges(data[self.input_key], self.apply_chat_template, self.tokenizer, self.max_length)
            # print(f"RANGES {response_ranges}")
        prompt, response, label = preprocess_data(
            data, self.input_template, self.input_key, self.output_key, self.label_key, self.apply_chat_template
        )
        prompt_token = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        # filter the sample whose length is greater than max_length (2 for answer length)
        if prompt_ids_len >= self.max_length - 2:
            prompt = None

        return {"prompt": prompt, "response": response, "label": label, "prompt_ids_len": prompt_ids_len, "response_ranges": response_ranges if self.multiturn else None}

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt, response = self.prompts[index], self.responses[index]
        text = prompt.rstrip("\n")
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
   
        # to avoid EOS_token truncation
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
            
        info = {
            "input_length": input_token["attention_mask"].int().sum().item(),
            "response_ranges": self.response_ranges[index] if self.multiturn else None,
        }
        
        # print(type(prompt), type(response), type(self.labels[index]), type(self.prompt_ids_lens[index]), type(info))
        # print(repr(prompt), repr(response), self.labels[index], self.prompt_ids_lens[index], info)
        return self.prompts[index], self.responses[index], self.labels[index], self.prompt_ids_lens[index], info

    def collate_fn(self, item_list):
        raise NotImplementedError
        def tokenizer(prompt, response):
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

            inputs["input_ids"][0][-1] = self.tokenizer.eos_token_id
            inputs["attention_mask"][0][-1] = True
            return inputs["input_ids"], inputs["attention_mask"]

        tot_ids, tot_masks, tot_labels, prompt_ids_lens, response_ranges = [], [], [], [], []
        for prompt, response, label, prompt_ids_len, response_range in item_list:
            input_ids, attention_mask = tokenizer(prompt, response)
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(label)
            prompt_ids_lens.append(prompt_ids_len)
            response_ranges.append(response_range)

        # add unmatched y'| x (used to estimate the KL divergence between policy and reference)
        for idx in range(len(item_list)):
            next_idx = (idx + 1) % len(item_list)
            input_ids, attention_mask = tokenizer(item_list[idx][0], item_list[next_idx][1])
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(-1)
            prompt_ids_lens.append(item_list[idx][3])
            response_ranges.append(item_list[idx][4])

        input_ids = zero_pad_sequences(tot_ids, side="right", value=self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(tot_masks, side="right")
        return input_ids, attention_mask, torch.LongTensor(tot_labels), prompt_ids_lens, response_ranges

    def packing_collate_fn(self, item_list):
        def tokenizer(prompt, response):
            if response:
                prompt = prompt + response
            text = prompt.rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs["input_ids"][0][-1] = self.tokenizer.eos_token_id
            inputs["attention_mask"][0][-1] = True
            return inputs["input_ids"], inputs["attention_mask"]

        packed_input_ids = []
        packed_attention_masks = []
        packed_labels = []
        prompt_ids_lens = []
        infos = {"input_length": [], "response_ranges": []}
        index = 1
        item_list_clone = copy.deepcopy(item_list)

        for prompt, response, label, prompt_ids_len, info in item_list:
            input_ids, attention_mask = tokenizer(prompt, response)
            # print(input_ids.shape, attention_mask.shape, repr(prompt))

            if self.multiple_of > 1:
                total_length = input_ids.size(-1)
                padding_len = (self.multiple_of - (total_length % self.multiple_of)) % self.multiple_of
                if padding_len > 0:
                    input_ids = F.pad(input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
                    attention_mask = F.pad(attention_mask, (0, padding_len), value=0)

            packed_input_ids.append(input_ids.flatten())
            
            attention_mask_flat = torch.full_like(input_ids.flatten(), index)
            if padding_len > 0:
                attention_mask_flat[-padding_len:] = 0
            packed_attention_masks.append(attention_mask_flat)
            
            packed_labels.append(label)
            prompt_ids_lens.append(prompt_ids_len)
            infos["input_length"].append(info["input_length"])
            if len(infos["response_ranges"]) >= 1:
                for i in range(len(info["response_ranges"])):
                    info["response_ranges"][i][0] += infos["response_ranges"][-1][-1][1]  # end_index of the last response of the last item
                    info["response_ranges"][i][1] += infos["response_ranges"][-1][-1][1]
            infos["response_ranges"].append(info["response_ranges"])
            index += 1

        # add unmatched y'| x (used to estimate the KL divergence between policy and reference)
        for idx in range(len(item_list_clone)):
            prompt, response, _, _, info = item_list_clone[idx]
            next_idx = (idx + 1) % len(item_list_clone)
            input_ids, attention_mask = tokenizer(item_list_clone[idx][0], item_list_clone[next_idx][1])

            if self.multiple_of > 1:
                total_length = input_ids.size(-1)
                padding_len = (self.multiple_of - (total_length % self.multiple_of)) % self.multiple_of
                if padding_len > 0:
                    input_ids = F.pad(input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
                    attention_mask = F.pad(attention_mask, (0, padding_len), value=0)

            packed_input_ids.append(input_ids.flatten())

            attention_mask_flat = torch.full_like(input_ids.flatten(), index)
            if padding_len > 0:
                attention_mask_flat[-padding_len:] = 0 
            packed_attention_masks.append(attention_mask_flat)
            
            packed_labels.append(-1)
            prompt_ids_lens.append(item_list_clone[idx][3])
            infos["input_length"].append(info["input_length"])
            if len(infos["response_ranges"]) >= 1:
                for i in range(len(info["response_ranges"])):
                    info["response_ranges"][i][0] += infos["response_ranges"][-1][-1][1]  # end_index of the last response of the last item
                    info["response_ranges"][i][1] += infos["response_ranges"][-1][-1][1]
            infos["response_ranges"].append(info["response_ranges"])
            index += 1

        infos["response_ranges"] = [item for sublist in infos["response_ranges"] for item in sublist]
        packed_input_ids = torch.stack(packed_input_ids, dim=0)
        packed_attention_masks = torch.stack(packed_attention_masks, dim=0)
        packed_labels = torch.LongTensor(packed_labels)

        # for idx, token in enumerate(packed_input_ids[0]):
        #     char = self.tokenizer.decode(token)
        #     print(f"At idx {idx}, Token: {token}, Character: {char}")
        # print(f"packed_input_ids: {packed_input_ids.shape}")
        # print(f"packed_attention_masks: {packed_attention_masks}")
        # print(f"packed_labels: {packed_labels}")
        # print(f"prompt_ids_lens: {prompt_ids_lens}")
        # print(f"infos: {infos['response_ranges']}")
        # exit()

        return packed_input_ids, packed_attention_masks, packed_labels, prompt_ids_lens, infos