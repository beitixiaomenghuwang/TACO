# collect representations for libero 10

output_dir="./representation_collection/openvla_libero/" # The path where you will store the collected features 
vla_path="openvla/openvla-7b-finetuned-libero-10" # Use the model offered by openvla on Huggingface
data_root_dir="your/path/to/openvla/modified_libero_rlds/"

python scripts/collect_inernal_representation/openvla_libero/collect.py \
  --vla_path $vla_path \
  --data_root_dir $data_root_dir \
  --dataset_name libero_10_no_noops \
  --run_root_dir ./temp \
  --adapter_tmp_dir ./temp \
  --batch_size 16 \
  --image_aug False \
  --output_dir $output_dir
