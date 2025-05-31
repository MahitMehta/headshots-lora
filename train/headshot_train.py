import subprocess
import os
import tempfile
import json
import sys

with open("headshot_train_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

if len(sys.argv) > 1 and sys.argv[1] == "--resume":
    config["resume"] = sys.argv[2]

base_dir = os.path.dirname(os.path.abspath(__file__))
train_script = os.path.join(base_dir, "kohya_ss", "sd-scripts", "train_network.py")

num_cpu_threads_per_process = config.get("num_cpu_threads_per_process", 2)
accelerate_cmd = [
    "accelerate",
    "launch",
    f"--num_cpu_threads_per_process={num_cpu_threads_per_process}",
]
if config.get("mixed_precision", "no") != "no":
    accelerate_cmd.append(f"--mixed_precision={config['mixed_precision']}")
if config.get("dynamo_backend", "no") != "no":
    accelerate_cmd.append(f"--dynamo_backend={config['dynamo_backend']}")
extra_accel_args = config.get("extra_accelerate_launch_args", "")
if extra_accel_args:
    accelerate_cmd.extend(extra_accel_args.split())
accelerate_cmd.append(train_script)

# Build train_network.py Arguments
train_args = []


def add_arg(name, value, is_flag=False, ignore_if_empty_str=True, positive_check=False):
    if is_flag:
        if value:
            train_args.append(name)
    elif positive_check:
        try:
            if value is not None and float(value) > 0:
                train_args.append(f"{name}={value}")
        except (ValueError, TypeError):
            pass
    elif value is not None:
        if isinstance(value, str) and ignore_if_empty_str and not value:
            return
        train_args.append(f"{name}={value}")


# Previous state
if config["resume"]:
    add_arg("--resume", config["resume"])

# Model and Paths
add_arg("--pretrained_model_name_or_path", config["pretrained_model_name_or_path"])
add_arg("--train_data_dir", config["train_data_dir"])
add_arg("--output_dir", config["output_dir"])
add_arg("--output_name", config["output_name"])
add_arg("--dataset_config", config.get("dataset_config", ""))
add_arg("--logging_dir", config.get("logging_dir", ""))
add_arg("--vae", config.get("vae", ""))

# Resolution and Bucketing
add_arg("--resolution", config["max_resolution"])

# Training Parameters
add_arg("--train_batch_size", config["train_batch_size"])
add_arg("--gradient_accumulation_steps", config["gradient_accumulation_steps"])
add_arg("--learning_rate", config["learning_rate"])
add_arg("--lr_scheduler", config["lr_scheduler"])
add_arg("--optimizer_type", config["optimizer"])

current_max_train_steps = config.get("max_train_steps", 0)
current_epoch = config.get(
    "epoch", 0
)  # epoch from JSON maps to max_train_epochs effectively
current_lr_warmup_steps = config.get("lr_warmup_steps", 0)
current_lr_warmup_ratio_perc = config.get("lr_warmup", 0)

if current_max_train_steps > 0:
    add_arg("--max_train_steps", current_max_train_steps)
    if current_lr_warmup_steps == 0 and current_lr_warmup_ratio_perc > 0:
        warmup_as_steps = int(
            current_max_train_steps * (current_lr_warmup_ratio_perc / 100)
        )
        if warmup_as_steps > 0:
            add_arg("--lr_warmup_steps", warmup_as_steps)
    elif current_lr_warmup_steps > 0:
        add_arg("--lr_warmup_steps", current_lr_warmup_steps)
elif current_epoch > 0:
    add_arg("--max_train_epochs", current_epoch)
    if current_lr_warmup_steps == 0 and current_lr_warmup_ratio_perc > 0:
        warmup_ratio = current_lr_warmup_ratio_perc / 100.0
        if warmup_ratio > 0:
            add_arg("--lr_warmup_ratio", warmup_ratio)
    elif current_lr_warmup_steps > 0:
        add_arg("--lr_warmup_steps", current_lr_warmup_steps)

add_arg("--lr_scheduler_args", config.get("lr_scheduler_args", ""))
if config.get("lr_scheduler_num_cycles", 0) > 0 and config.get("lr_scheduler") in [
    "cosine_with_restarts",
    "polynomial_with_restarts",
    "cosine",
]:
    add_arg("--lr_scheduler_num_cycles", config["lr_scheduler_num_cycles"])
if (
    config.get("lr_scheduler_power", 1) != 1
    and config.get("lr_scheduler") == "polynomial"
):
    add_arg("--lr_scheduler_power", config["lr_scheduler_power"])
optimizer_args_str = config.get("optimizer_args", "")
if optimizer_args_str:
    train_args.extend(optimizer_args_str.split())

# Network (LoRA) settings
if config.get("LoRA_type") == "Standard":
    add_arg("--network_module", "networks.lora")
add_arg("--network_dim", config["network_dim"])
add_arg("--network_alpha", config["network_alpha"])

lora_network_args_list = []
if config.get("block_dims"):
    lora_network_args_list.append(f"network_block_dims={config.get('block_dims')}")
if config.get("block_alphas"):
    lora_network_args_list.append(f"network_block_alphas={config.get('block_alphas')}")
if config.get("block_lr_zero_threshold"):
    lora_network_args_list.append(
        f"block_lr_zero_threshold={config.get('block_lr_zero_threshold')}"
    )
if config.get("down_lr_weight"):
    lora_network_args_list.append(f"down_lr_weight={config.get('down_lr_weight')}")
if config.get("mid_lr_weight"):
    lora_network_args_list.append(f"mid_lr_weight={config.get('mid_lr_weight')}")
if config.get("up_lr_weight"):
    lora_network_args_list.append(f"up_lr_weight={config.get('up_lr_weight')}")
if config.get("additional_parameters"):
    lora_network_args_list.append(
        config.get("additional_parameters")
    )  # For raw passthrough
if lora_network_args_list:
    train_args.append(f"--network_args {' '.join(lora_network_args_list)}")

# Precision and Saving
add_arg("--mixed_precision", config.get("mixed_precision", "no"))
add_arg("--save_precision", config["save_precision"])
add_arg("--save_model_as", config["save_model_as"])
add_arg(
    "--save_every_n_epochs", config.get("save_every_n_epochs", 0), positive_check=True
)
add_arg(
    "--save_every_n_steps", config.get("save_every_n_steps", 0), positive_check=True
)

# Other Training Options
add_arg(
    "--gradient_checkpointing",
    config.get("gradient_checkpointing", False),
    is_flag=True,
)
add_arg("--cache_latents", config.get("cache_latents", False), is_flag=True)
if config.get("cache_latents", False):
    add_arg(
        "--cache_latents_to_disk",
        config.get("cache_latents_to_disk", False),
        is_flag=True,
    )

add_arg("--clip_skip", config.get("clip_skip", 0))
add_arg("--max_token_length", 150)
add_arg("--seed", config.get("seed"))
add_arg("--prior_loss_weight", config.get("prior_loss_weight", 1.0))
add_arg("--max_grad_norm", config.get("max_grad_norm", 1.0))

if config.get("xformers") == "xformers":
    add_arg("--xformers", True, is_flag=True)
elif config.get("xformers") == "sdpa":
    add_arg("--sdpa", True, is_flag=True)

add_arg(
    "--persistent_data_loader_workers",
    config.get("persistent_data_loader_workers", False),
    is_flag=True,
)
add_arg("--max_data_loader_n_workers", config.get("max_data_loader_n_workers", 0))

# Captioning
add_arg("--caption_extension", config.get("caption_extension", ".txt"))
add_arg("--shuffle_caption", config.get("shuffle_caption", False), is_flag=True)
add_arg(
    "--caption_dropout_every_n_epochs",
    config.get("caption_dropout_every_n_epochs", 0),
    positive_check=True,
)
add_arg(
    "--caption_dropout_rate", config.get("caption_dropout_rate", 0), positive_check=True
)
add_arg("--weighted_captions", config.get("weighted_captions", False), is_flag=True)

# Augmentations
add_arg("--flip_aug", config.get("flip_aug", False), is_flag=True)
add_arg("--color_aug", config.get("color_aug", False), is_flag=True)
add_arg("--random_crop", config.get("random_crop", False), is_flag=True)

add_arg("--min_snr_gamma", config.get("min_snr_gamma", 0))
add_arg(
    "--debiased_estimation_loss",
    config.get("debiased_estimation_loss", False),
    is_flag=True,
)
add_arg("--loss_type", config.get("loss_type", "l2"))
add_arg("--huber_schedule", config.get("huber_schedule", "snr"))
add_arg("--huber_c", config.get("huber_c", 0.1))

# Sampling (for previews)
sample_prompts_path = None
if (
    config.get("sample_every_n_steps", 0) > 0
    or config.get("sample_every_n_epochs", 0) > 0
):
    prompts_str = config.get("sample_prompts", "")
    if prompts_str:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as tmpfile:
            prompts_cleaned = [
                p.strip()
                for p_line in prompts_str.split(",")
                for p in p_line.split(";")
                if p.strip()
            ]
            tmpfile.write("\n---\n".join(prompts_cleaned))
            sample_prompts_path = tmpfile.name
        add_arg("--sample_prompts", sample_prompts_path)
        add_arg("--sample_sampler", config.get("sample_sampler", "euler_a"))
    add_arg(
        "--sample_every_n_steps",
        config.get("sample_every_n_steps", 0),
        positive_check=True,
    )
    add_arg(
        "--sample_every_n_epochs",
        config.get("sample_every_n_epochs", 0),
        positive_check=True,
    )

# Other flags
add_arg("--full_fp16", config.get("full_fp16", False), is_flag=True)
add_arg("--full_bf16", config.get("full_bf16", False), is_flag=True)
add_arg("--fp8_base", config.get("fp8_base", False), is_flag=True)
add_arg("--v2", config.get("v2", False), is_flag=True)
add_arg("--v_parameterization", config.get("v_parameterization", False), is_flag=True)
add_arg(
    "--stop_text_encoder_training",
    config.get("stop_text_encoder_training", 0),
    positive_check=True,
)
add_arg("--train_norm", config.get("train_norm", False), is_flag=True)
add_arg(
    "--scale_weight_norms", config.get("scale_weight_norms", 0), positive_check=True
)

if config.get("save_state", False):
    add_arg("--save_state", True, is_flag=True)
# save_state_on_train_end is implicitly True if save_state is True in Kohya's scripts.

log_with_val = config.get("log_with", "")
add_arg("--log_with", log_with_val)
if log_with_val == "wandb":
    add_arg("--wandb_api_key", config.get("wandb_api_key", ""))
    add_arg("--wandb_run_name", config.get("wandb_run_name", ""))

add_arg("--text_encoder_lr", config.get("text_encoder_lr", 0), positive_check=True)
add_arg("--unet_lr", config.get("unet_lr", 0), positive_check=True)
add_arg("--module_dropout", config.get("module_dropout", 0), positive_check=True)
add_arg("--vae_batch_size", config.get("vae_batch_size", 0), positive_check=True)
add_arg("--max_timestep", config.get("max_timestep", 1000))
add_arg("--min_timestep", config.get("min_timestep", 0))

if config.get("masked_loss"):
    add_arg("--masked_loss", config["masked_loss"], is_flag=True)

# --- Execute Command ---
final_command = accelerate_cmd + [arg for arg in train_args if arg]

print("Generated command:")
command_str_parts = []
for part in final_command:
    if " " in part and not ("=" in part and part.startswith("--network_args")):
        command_str_parts.append(f'"{part}"')
    else:
        command_str_parts.append(part)
print(" ".join(command_str_parts))
print("\n")

try:
    print(f"⏳ Starting Kohya's SS LoRA training...")
    process = subprocess.Popen(
        final_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    if rc == 0:
        print(f"✅ Training completed successfully.")
    else:
        print(f"❌ Training process exited with an error code: {rc}")
except FileNotFoundError as e:
    print(f"❌ Error: Could not find an executable: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
finally:
    if sample_prompts_path and os.path.exists(sample_prompts_path):
        os.remove(sample_prompts_path)
        print(f"🧹 Cleaned up temporary prompt file: {sample_prompts_path}")
print("Script finished.")
