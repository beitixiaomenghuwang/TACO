# Please modify "policy_path", "tag", "cfn_ckpt_path" !!!!

# your cfn_ckpt_path !
# Here, we recommend using the absolute path.
export cfn_ckpt_path="your/path/to/cfn_ckpt.pt" 

cd ./third_party/Robotwin

policy_name=pi05
policy_path=your/path/to/trained/pi0.5 # model report id, or trained lerobot model checkpoint, e.g. ./outputs/pi05_training/checkpoints/100000/pretrained_model
task_config=demo_clean
seed=1
task_name=adjust_bottle
tag=test
# output dir is "./third_party/Robotwin/eval_result/{tag}"

cd ./third_party/Robotwin

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_lerobot_torch_pi05_taco.py \
    --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --tag ${tag} \
    --policy_path $policy_path



