import os
import subprocess
import shlex

# Set base directory to the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to BASE_DIR
KOHYA_SS_SCRIPT_PATH = os.path.join(BASE_DIR, "kohya_ss", "sd-scripts", "train_network.py")
DATASET_CONFIG_PATH = os.path.join(BASE_DIR, "dataset_config.toml")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SAMPLE_PROMPTS_FILE = os.path.join(OUTPUT_DIR, "sample", "prompt.txt")

# Training parameters
PRETRAINED_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_NAME = "headshot_inpainting_lora_v1"
NETWORK_DIM = 4
NETWORK_ALPHA = 4
LEARNING_RATE = "0.0001"
TEXT_ENCODER_LR = "2e-5"
TRAIN_BATCH_SIZE = 1
MAX_TRAIN_STEPS = 550
OPTIMIZER_TYPE = "AdamW8bit"
LR_SCHEDULER = "cosine"
LR_SCHEDULER_NUM_CYCLES = 1
MIXED_PRECISION = "fp16"
SAVE_MODEL_AS = "safetensors"
SAVE_PRECISION = "fp16"
SAVE_EVERY_N_STEPS = 100
ENABLE_XFORMERS = True
CACHE_LATENTS = True

# Construct the command
command = [
    "accelerate", "launch", KOHYA_SS_SCRIPT_PATH,
    f"--pretrained_model_name_or_path={PRETRAINED_MODEL}",
    f"--dataset_config={DATASET_CONFIG_PATH}",
    f"--output_dir={OUTPUT_DIR}",
    f"--output_name={OUTPUT_NAME}",
    "--network_module=networks.lora",
    f"--network_dim={NETWORK_DIM}",
    f"--network_alpha={NETWORK_ALPHA}",
    f"--train_batch_size={TRAIN_BATCH_SIZE}",
    f"--learning_rate={LEARNING_RATE}",
    f"--unet_lr={LEARNING_RATE}",
    f"--text_encoder_lr={TEXT_ENCODER_LR}",
    f"--mixed_precision={MIXED_PRECISION}",
    f"--save_model_as={SAVE_MODEL_AS}",
    f"--save_precision={SAVE_PRECISION}",
    f"--optimizer_type={OPTIMIZER_TYPE}",
    f"--lr_scheduler={LR_SCHEDULER}",
    f"--lr_scheduler_num_cycles={LR_SCHEDULER_NUM_CYCLES}",
    f"--max_train_steps={MAX_TRAIN_STEPS}",
    "--masked_loss"
]

# Add sample prompts if the file exists
if os.path.exists(SAMPLE_PROMPTS_FILE):
    command.extend([
        "--sample_sampler=euler_a",
        f"--sample_prompts={SAMPLE_PROMPTS_FILE}"
    ])
else:
    print(f"Warning: Sample prompts file not found at {SAMPLE_PROMPTS_FILE}, skipping sample generation during training.")

# Add optional flags
if ENABLE_XFORMERS:
    command.append("--xformers")

if CACHE_LATENTS:
    command.append("--cache_latents")

# Print the constructed command
print("Generated command:")
print(" ".join(shlex.quote(str(arg)) for arg in command))

# Execute the command
print("\nStarting training process...")
try:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    
    # Stream output
    if process.stdout:
        for line in process.stdout:
            print(line, end='')
    
    process.wait()  # Wait for the process to complete

    if process.returncode == 0:
        print("\nTraining completed successfully.")
    else:
        print(f"\nTraining failed with error code: {process.returncode}")

except FileNotFoundError:
    print(f"Error: 'accelerate' command not found. Make sure accelerate is installed and in your PATH.")
    print(f"Attempted to run from script path: {KOHYA_SS_SCRIPT_PATH}")
except Exception as e:
    print(f"An error occurred while trying to run the training command: {e}")
