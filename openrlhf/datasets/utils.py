import torch
import torch.nn.functional as F


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def exist_and_not_none(d, key):
    return key in d and not d[key] is None

def get_ranges(data, apply_chat_template, tokenizer, max_length):
    ranges = []
    for idx, message in enumerate(data):
        if message['role'] == 'assistant':
            prompt = apply_chat_template(data[: idx], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[: idx + 1], tokenize=False)[len(prompt):]
            start_idx = tokenizer(
                prompt,
                max_length=max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )["attention_mask"].int().sum().item()
            
            end_idx = start_idx + tokenizer(
                response,
                max_length=max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )["attention_mask"].int().sum().item() - 1
            
            ranges.append((start_idx, end_idx)) # left close right open
            
    return ranges