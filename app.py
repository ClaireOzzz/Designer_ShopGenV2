import gradio as gr
import os
import subprocess
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN")


def set_accelerate_default_config():
    try:
        subprocess.run(["accelerate", "config", "default"], check=True)
        print("Accelerate default config set successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def train_dreambooth_lora_sdxl(instance_data_dir, lora_trained_xl_folder, instance_prompt, max_train_steps, checkpoint_steps):
    
    script_filename = "train_dreambooth_lora_sdxl.py"  # Assuming it's in the same folder

    command = [
        "accelerate",
        "launch",
        script_filename,  # Use the local script
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
        f"--instance_data_dir={instance_data_dir}",
        f"--output_dir={lora_trained_xl_folder}",
        "--mixed_precision=fp16",
        f"--instance_prompt={instance_prompt}",
        "--resolution=1024",
        "--train_batch_size=2",
        "--gradient_accumulation_steps=2",
        "--gradient_checkpointing",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--enable_xformers_memory_efficient_attention",
        "--mixed_precision=fp16",
        "--use_8bit_adam",
        f"--max_train_steps={max_train_steps}",
        f"--checkpointing_steps={checkpoint_steps}",
        "--seed=0",
        "--push_to_hub",
        f"--hub_token={hf_token}"
    ]

    try:
        subprocess.run(command, check=True)
        print("Training is finished!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def main(dataset_id, 
         lora_trained_xl_folder,
         instance_prompt,
         max_train_steps,
         checkpoint_steps):

    dataset_repo = dataset_id

    # Automatically set local_dir based on the last part of dataset_repo
    repo_parts = dataset_repo.split("/")
    local_dir = f"./{repo_parts[-1]}"  # Use the last part of the split

    # Check if the directory exists and create it if necessary
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    gr.Info("Downloading dataset ...")
    
    snapshot_download(
        dataset_repo,
        local_dir=local_dir,
        repo_type="dataset",
        ignore_patterns=".gitattributes",
        token=hf_token
    )

    set_accelerate_default_config()

    gr.Info("Training begins ...")

    instance_data_dir = repo_parts[-1]
    train_dreambooth_lora_sdxl(instance_data_dir, lora_trained_xl_folder, instance_prompt, max_train_steps, checkpoint_steps)

    return f"Done, your trained model has been stored in your models library: your_user_name/{lora-trained-xl-folder}"

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            dataset_id = gr.Textbox(label="Dataset ID", info="use one of your previously uploaded datasets on your HF profile", placeholder="diffusers/dog-example")
            instance_prompt = gr.Textbox(label="Concept prompt", info="concept prompt - use a unique, made up word to avoid collisions")
        
        with gr.Row():
            model_output_folder = gr.Textbox(label="Output model folder name", placeholder="lora-trained-xl-folder")
            max_train_steps = gr.Number(label="Max Training Steps", value=500)
            checkpoint_steps = gr.Number(label="Checkpoints Steps", value=100)
        train_button = gr.Button("Train !")
        status = gr.Textbox(labe="Training status")

    train_button.click(
        fn = main,
        inputs = [
            dataset_id,
            model_output_folder,
            instance_prompt,
            max_train_steps,
            checkpoint_steps
        ],
        outputs = [status]
    )

demo.queue().launch()