DPO, KTO 需要保证 prompt 在 chosen_key 和 rejected_key 里同时出现
{
    "prompt": [{"role": "user": "content": "How are you"}],
    "chosen": [{"role": "user": "content": "How are you"},{"role": "assistant": "content": "How are you1"}],
    "rejected": [{"role": "user": "content": "How are you"},{"role": "assistant": "content": "How are you2"}],
}