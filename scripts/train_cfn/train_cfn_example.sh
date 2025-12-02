# Please modify "feature_dir", "output_dir", "input_dim" !!!!!

# Path where the representation is stored, e.g.:
feature_dir="./representation_collection/pi05_libero/feature.pt"

output_dir="./cfn_output/"
input_dim=1024 # feature dim, 4096 if openvla, 1024 if pi0.5
multi_feature_file=False

python scripts/train_cfn/train_cfn.py \
  --output_dir $output_dir \
  --batch_size 512 \
  --num_workers 16 \
  --epochs 16 \
  --save_freq 4 \
  --feature_dir $feature_dir \
  --multi_feature_file $multi_feature_file \
  --input_dim $input_dim \
  --cfn_output_dim 20 \
  

