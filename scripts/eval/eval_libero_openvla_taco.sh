
# Plz modify "cfn_ckpt_path", "output_dir" !!!!

export cfn_ckpt_path="./cfn_output/model_epoch16.pt"  # your cfn_ckpt_path !

vla_path="openvla/openvla-7b-finetuned-libero-10" # Needs to be consistent with the representation collection
output_dir="./output/eval_openvla_libero/"

python third_party/openvla/experiments/robot/libero/run_libero_eval_openvla_cfntts.py \
  --model_family openvla \
  --pretrained_checkpoint $vla_path \
  --task_suite_name libero_10 \
  --center_crop True \
  --local_log_dir $output_dir
  