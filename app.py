import gradio as gr
import torch
import os
import shutil
import requests
import subprocess
from subprocess import getoutput
from huggingface_hub import snapshot_download, HfApi, create_repo
api = HfApi()

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

def load_images_to_dataset(images, dataset_name):

    if dataset_name == "":
        raise gr.Error("You forgot to name your new dataset. ")

    # Create the directory if it doesn't exist
    my_working_directory = f"my_working_directory_for_{dataset_name}"
    if not os.path.exists(my_working_directory):
        os.makedirs(my_working_directory)

    # Assuming 'images' is a list of image file paths
    for idx, image in enumerate(images):
        # Get the base file name (without path) from the original location
        image_name = os.path.basename(image.name)
    
        # Construct the destination path in the working directory
        destination_path = os.path.join(my_working_directory, image_name)
    
        # Copy the image from the original location to the working directory
        shutil.copy(image.name, destination_path)
    
        # Print the image name and its corresponding save path
        print(f"Image {idx + 1}: {image_name} copied to {destination_path}")
   
    path_to_folder = my_working_directory
    your_username = api.whoami(token=hf_token)["name"]
    repo_id = f"{your_username}/{dataset_name}"
    create_repo(repo_id=repo_id, repo_type="dataset", private=True, token=hf_token)
    
    api.upload_folder(
        folder_path=path_to_folder,
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )

    return "Done, your dataset is ready and loaded for the training step!", repo_id

def swap_hardware(hf_token, hardware="cpu-basic"):
    hardware_url = f"https://huggingface.co/spaces/{os.environ['SPACE_ID']}/hardware"
    headers = { "authorization" : f"Bearer {hf_token}"}
    body = {'flavor': hardware}
    requests.post(hardware_url, json = body, headers=headers)

def swap_sleep_time(hf_token,sleep_time):
    sleep_time_url = f"https://huggingface.co/api/spaces/{os.environ['SPACE_ID']}/sleeptime"
    headers = { "authorization" : f"Bearer {hf_token}"}
    body = {'seconds':sleep_time}
    requests.post(sleep_time_url,json=body,headers=headers)

def get_sleep_time(hf_token):
    sleep_time_url = f"https://huggingface.co/api/spaces/{os.environ['SPACE_ID']}"
    headers = { "authorization" : f"Bearer {hf_token}"}
    response = requests.get(sleep_time_url,headers=headers)
    try:
        gcTimeout = response.json()['runtime']['gcTimeout']
    except:
        gcTimeout = None
    return gcTimeout

def write_to_community(title, description,hf_token): 
    
    api.create_discussion(repo_id=os.environ['SPACE_ID'], title=title, description=description,repo_type="space", token=hf_token)


def set_accelerate_default_config():
    try:
        subprocess.run(["accelerate", "config", "default"], check=True)
        print("Accelerate default config set successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def train_dreambooth_lora_sdxl(dataset_id, instance_data_dir, lora_trained_xl_folder, instance_prompt, max_train_steps, checkpoint_steps, remove_gpu):
    
    script_filename = "train_dreambooth_lora_sdxl.py"  # Assuming it's in the same folder

    command = [
        "accelerate",
        "launch",
        script_filename,  # Use the local script
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
        f"--dataset_id={dataset_id}",
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
        #swap_hardware(hf_token, "cpu-basic")
        #write_to_community(title,description,hf_token)

def main(dataset_id, 
         lora_trained_xl_folder,
         instance_prompt,
         max_train_steps,
         checkpoint_steps,
         remove_gpu):

    
    if is_shared_ui:
        raise gr.Error("This Space only works in duplicated instances")

    if not is_gpu_associated:
        raise gr.Error("Please associate a T4 or A10G GPU for this Space")

    if dataset_id == "":
        raise gr.Error("You forgot to specify an image dataset")

    if instance_prompt == "":
        raise gr.Error("You forgot to specify a concept prompt")

    if lora_trained_xl_folder == "":
        raise gr.Error("You forgot to name the output folder for your model")

    sleep_time = get_sleep_time(hf_token)
    if sleep_time:
        swap_sleep_time(hf_token, -1)

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
    train_dreambooth_lora_sdxl(dataset_id, instance_data_dir, lora_trained_xl_folder, instance_prompt, max_train_steps, checkpoint_steps, remove_gpu)
    
    your_username = api.whoami(token=hf_token)["name"]
    return f"Done, your trained model has been stored in your models library: {your_username}/{lora_trained_xl_folder}"

css="""
#col-container {max-width: 780px; margin-left: auto; margin-right: auto;}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        if is_shared_ui:
            top_description = gr.HTML(f'''
                <div class="gr-prose">
                <h2>Attention - This Space doesn't work in this shared UI</h2>
                <p>For it to work, you can duplicate the Space and run it on your own profile using a (paid) private T4-small or A10G-small GPU for training. A T4 costs US$0.60/h, so it should cost < US$1 to train most models using default settings with it!&nbsp;&nbsp;<a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></p>
                </div>
            ''')
        else:
            if(is_gpu_associated):
                top_description = gr.HTML(f'''
                        <div class="gr-prose">
                        <h2>You have successfully associated a {which_gpu} GPU to the SD-XL Dreambooth LoRa Training Space 🎉</h2>
                        <p>You can now train your model! You will be billed by the minute from when you activated the GPU until when it is turned it off.</p> 
                        </div>
                ''')
            else:
                top_description = gr.HTML(f'''
                        <div class="gr-prose">
                        <h2>You have successfully duplicated the SD-XL Dreambooth LoRa Training Space 🎉</h2>
                        <p>There's only one step left before you can train your model: <a href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}/settings" style="text-decoration: underline" target="_blank">attribute a <b>T4-small or A10G-small GPU</b> to it (via the Settings tab)</a> and run the training below. You will be billed by the minute from when you activate the GPU until when it is turned it off.</p> 
                        </div>
                ''')
        gr.Markdown("# SD-XL Dreambooth LoRa Training UI 💭")
        gr.Markdown("## Drop your training images (optional)")
        gr.Markdown("Use this step to upload your training images. If you already have a dataset stored on your HF profile, you can skip this step, and provide your dataset ID in the training `Datased ID` input below.")
        images = gr.File(file_types=["image"], label="Upload your images", file_count="multiple", interactive=True, visible=True)
        with gr.Row():
            new_dataset_name = gr.Textbox(label="Set new dataset name", placeholder="e.g.: my_awesome_dataset")
            load_btn = gr.Button("Load images to new dataset")
        dataset_status = gr.Textbox(label="dataset status")
        gr.Markdown("## Training ")
        gr.Markdown("You can use an existing image dataset, find a dataset example here: [https://huggingface.co/datasets/diffusers/dog-example](https://huggingface.co/datasets/diffusers/dog-example) ;)")
        with gr.Row():
            dataset_id = gr.Textbox(label="Dataset ID", info="use one of your previously uploaded image datasets on your HF profile", placeholder="diffusers/dog-example")
            instance_prompt = gr.Textbox(label="Concept prompt", info="concept prompt - use a unique, made up word to avoid collisions")
        
        with gr.Row():
            model_output_folder = gr.Textbox(label="Output model folder name", placeholder="lora-trained-xl-folder")
            max_train_steps = gr.Number(label="Max Training Steps", value=500, precision=0, step=10)
            checkpoint_steps = gr.Number(label="Checkpoints Steps", value=100, precision=0, step=10)
        remove_gpu = gr.Checkbox(label="Remove GPU After Training", value=True)
        train_button = gr.Button("Train !")

        
        train_status = gr.Textbox(label="Training status")
    
    load_btn.click(
        fn = load_images_to_dataset,
        inputs = [images, new_dataset_name],
        outputs = [dataset_status, dataset_id]
    )
    
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
        outputs = [train_status]
    )

demo.queue(default_enabled=False).launch(debug=True)