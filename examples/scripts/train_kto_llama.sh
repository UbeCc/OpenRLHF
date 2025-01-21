set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_kto \
   --save_path /workspace/haoran/models/test \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --max_samples 64 \
   --pretrain /workspace/haoran/models/Qwen/Qwen2.5-Coder-1.5B-Instruct/ \
   --bf16 \
   --max_epochs 1 \
   --max_len 1024 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset /workspace/haoran/OpenRLHF/datasets/kto-demo.json \
   --input_key input \
   --label_key reward \
   --flash_attn \
   --beta 0.1 \
   --max_samples 1024 \
   --packing_samples \
   --multiturn \
   --ring_attn_size 2 \
   --ring_head_stride 2 \
   --apply_chat_template \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
