import gradio as gr
import torch
import os
import subprocess
from subprocess import getoutput
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN_WITH_WRITE_PERMISSION")

is_shared_ui = True if "fffiloni/train-dreambooth-lora-sdxl" in os.environ['SPACE_ID'] else False


is_gpu_associated = torch.cuda.is_available()
if is_gpu_associated:
    gpu_info = getoutput('nvidia-smi')
    if("A10G" in gpu_info):
        which_gpu = "A10G"
    elif("T4" in gpu_info):
        which_gpu = "T4"
    else:
        which_gpu = "CPU"

def set_accelerate_default_config():
    try:
        subprocess.run(["accelerate", "config", "default"], check=True)
        print("Accelerate default config set successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def train_dreambooth_lora_sdxl(instance_data_dir, lora_trained_xl_folder, instance_prompt, max_train_steps, checkpoint_steps, remove_gpu):
    
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
        if remove_gpu:
            swap_hardware(hf_token, "cpu-basic")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        
        title="There was an error on during your training"
        description=f'''
        Unfortunately there was an error during training your {lora_trained_xl_folder} model. 
        Please check it out below. Feel free to report this issue to [SD-XL Dreambooth LoRa Training](https://huggingface.co/spaces/fffiloni/train-dreambooth-lora-sdxl): 
        ```
        {str(e)}
        ```
        '''
        swap_hardware(hf_token, "cpu-basic")
        write_to_community(title,description,hf_token)

def main(dataset_id, 
         lora_trained_xl_folder,
         instance_prompt,
         max_train_steps,
         checkpoint_steps,
         remove_gpu):

    if dataset_id == "":
        raise gr.Error("You forgot to specify an image dataset")

    if instance_prompt == "":
        raise gr.Error("You forgot to specify a concept prompt")

    if lora_trained_xl_folder == "":
        raise gr.Error("You forgot to name the output folder for your model")
    
    if is_shared_ui:
        raise gr.Error("This Space only works in duplicated instances")

    if not is_gpu_associated:
        raise gr.Error("Please associate a T4 or A10G GPU for this Space")

    gr.Warning("Training is ongoing ⌛... You can close this tab if you like or just wait.")
    gr.Warning("If you did not check the `Remove GPU After training`, don't forget to remove the GPU attribution after you are done. ")
        
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
    train_dreambooth_lora_sdxl(instance_data_dir, lora_trained_xl_folder, instance_prompt, max_train_steps, checkpoint_steps, remove_gpu)

    return f"Done, your trained model has been stored in your models library: your_user_name/{lora-trained-xl-folder}"

css="""
#col-container {max-width: 780px; margin-left: auto; margin-right: auto;}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        if is_shared_ui:
            top_description = gr.HTML(f'''
                <div class="gr-prose" style="max-width: 80%">
                <h2>Attention - This Space doesn't work in this shared UI</h2>
                <p>For it to work, you can duplicate the Space and run it on your own profile using a (paid) private T4-small or A10G-small GPU for training. A T4 costs US$0.60/h, so it should cost < US$1 to train most models using default settings with it!&nbsp;&nbsp;<a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></p>
                </div>
            ''')
        else:
            if(is_gpu_associated):
                top_description = gr.HTML(f'''
                        <div class="gr-prose" style="max-width: 80%">
                        <h2>You have successfully associated a {which_gpu} GPU to the SD-XL Dreambooth LoRa Training Space 🎉</h2>
                        <p>You can now train your model! You will be billed by the minute from when you activated the GPU until when it is turned it off.</p> 
                        </div>
                ''')
            else:
                top_description = gr.HTML(f'''
                        <div class="gr-prose" style="max-width: 80%">
                        <h2>You have successfully duplicated the SD-XL Dreambooth LoRa Training Space 🎉</h2>
                        <p>There's only one step left before you can train your model: <a href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}/settings" style="text-decoration: underline" target="_blank">attribute a <b>T4-small or A10G-small GPU</b> to it (via the Settings tab)</a> and run the training below. You will be billed by the minute from when you activate the GPU until when it is turned it off.</p> 
                        </div>
                ''')
        gr.Markdown("# SD-XL Dreambooth LoRa Training UI 💭")
        with gr.Row():
            dataset_id = gr.Textbox(label="Dataset ID", info="use one of your previously uploaded image datasets on your HF profile", placeholder="diffusers/dog-example")
            instance_prompt = gr.Textbox(label="Concept prompt", info="concept prompt - use a unique, made up word to avoid collisions")
        
        with gr.Row():
            model_output_folder = gr.Textbox(label="Output model folder name", placeholder="lora-trained-xl-folder")
            max_train_steps = gr.Number(label="Max Training Steps", value=500)
            checkpoint_steps = gr.Number(label="Checkpoints Steps", value=100)
            remove_gpu = gr.Checkbox(label="Remove GPU After Training", value=True)
        train_button = gr.Button("Train !")

        
        status = gr.Textbox(label="Training status")

    train_button.click(
        fn = main,
        inputs = [
            dataset_id,
            model_output_folder,
            instance_prompt,
            max_train_steps,
            checkpoint_steps,
            remove_gpu
        ],
        outputs = [status]
    )

demo.queue().launch()