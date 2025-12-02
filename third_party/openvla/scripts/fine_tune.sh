
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /path/to/openvla/modified_libero_rlds/ \
  --dataset_name libero_10_no_noops \
  --run_root_dir /output \
  --adapter_tmp_dir /temp \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 20000