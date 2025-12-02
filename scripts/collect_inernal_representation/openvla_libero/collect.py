
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
from accelerate import PartialState
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform_yang, RLDSDataset

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from PIL import Image
import numpy as np
from tqdm import tqdm

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on
    output_dir: str = "./representation_collection/openvla_libero/"


@draccus.wrap()
def collect(cfg: FinetuneConfig) -> None:
    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Quantization Config 
    quantization_config = None

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    vla = vla.to(device_id)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform_yang(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    features_good = []
    # Check whether the 'vla_dataset' is capable of being iterated infinitely !!!!!
    # It should stop when an iteration is completed.
    for batch in tqdm(vla_dataset, desc="Processing dataset"):
        task_label = batch['task']['language_instruction']
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        
        image = Image.fromarray(batch["observation"]["image_primary"][0])
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

        features = []
        dises = []
        for i in range(16): 
            actions, norm_action, hidden_states = vla.predict_action_n_get_feature(
                **inputs, unnorm_key="libero_10", 
                do_sample=True,
                temperature = 1, 
                output_hidden_states=True,   # note
                return_dict_in_generate=True, # 
            )

            last_token_hidden = hidden_states[-1][-1][0, -1, :]
            feature = last_token_hidden.cpu()
            features.append(feature)

            dis = torch.tensor(np.linalg.norm(norm_action - batch["action"])).cpu()
            dises.append(dis)
        
        features = torch.stack(features)
        dises = torch.stack(dises)
        min_index = torch.argmin(dises)
        features_good.append(features[min_index])
    
    try: 
        save_path = cfg.output_dir
        features_good = torch.stack(features_good)
        os.makedirs(save_path, exist_ok=True)
        torch.save(features_good, f"{save_path}/feature.pt")
    except Exception:
        print("We got an error, but you can manually save the 'features_good' data")
        import ipdb;ipdb.set_trace()
        print()

    print("very good!")
    print(f"The feature has been saved at {save_path}")


if __name__ == "__main__":
    collect()
