set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
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
   --beta 0.1 \
   --dataset /workspace/haoran/OpenRLHF/datasets/dpo-demo.json \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --multiturn \
   --ring_attn_size 2 \
   --ring_head_stride 2 \
   --packing_samples \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
