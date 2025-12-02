# Please modify "cfn_ckpt_path", "output_dir" !!!

export TOKENIZERS_PARALLELISM=false
export cfn_ckpt_path="./cfn_output/model_epoch16.pt" # your cfn_ckpt_path !

output_dir="./output/eval_pi05_libero/"
policy_path="lerobot/pi05_libero_finetuned" # provided by lerobot

python ./third_party/lerobot/src/lerobot/scripts/lerobot_eval_tts.py \
  --output_dir=$output_dir \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --policy.path=$policy_path \
  --policy.n_action_steps=10 \
  --env.max_parallel_tasks=1


