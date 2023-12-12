import gradio as gr
from huggingface_hub import login, HfFileSystem, HfApi, ModelCard
import os
import spaces
import random
import torch

is_shared_ui = False

hf_token = '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'
login(token=hf_token)

fs = HfFileSystem(token=hf_token)
api = HfApi()

device="cuda" if torch.cuda.is_available() else "cpu"

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

def check_use_custom_or_no(value):
    if value is True:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def get_files(file_paths):
    last_files = {}  # Dictionary to store the last file for each path

    for file_path in file_paths:
        # Split the file path into directory and file components
        directory, file_name = file_path.rsplit('/', 1)
    
        # Update the last file for the current path
        last_files[directory] = file_name
    
    # Extract the last files from the dictionary
    result = list(last_files.values())

    return result

def load_model(model_name):

    if model_name == "":
        gr.Warning("If you want to use a private model, you need to duplicate this space on your personal account.")
        raise gr.Error("You forgot to define Model ID.")

    # Get instance_prompt a.k.a trigger word
    card = ModelCard.load(model_name)
    repo_data = card.data.to_dict()
    instance_prompt = repo_data.get("instance_prompt")

    if instance_prompt is not None:
        print(f"Trigger word: {instance_prompt}")
    else:
        instance_prompt = "no trigger word needed"
        print(f"Trigger word: no trigger word needed")

    # List all ".safetensors" files in repo
    sfts_available_files = fs.glob(f"{model_name}/*safetensors")
    sfts_available_files = get_files(sfts_available_files)

    if sfts_available_files == []:
        sfts_available_files = ["NO SAFETENSORS FILE"]

    print(f"Safetensors available: {sfts_available_files}")

    return model_name, "Model Ready", gr.update(choices=sfts_available_files, value=sfts_available_files[0], visible=True), gr.update(value=instance_prompt, visible=True)

def custom_model_changed(model_name, previous_model):
    if model_name == "" and previous_model == "" :
        status_message = ""      
    elif model_name != previous_model:
        status_message = "model changed, please reload before any new run"
    else:
        status_message = "model ready"
    return status_message

def resize_image(input_path, output_path, target_height):
    # Open the input image
    img = Image.open(input_path)

    # Calculate the aspect ratio of the original image
    original_width, original_height = img.size
    original_aspect_ratio = original_width / original_height

    # Calculate the new width while maintaining the aspect ratio and the target height
    new_width = int(target_height * original_aspect_ratio)

    # Resize the image while maintaining the aspect ratio and fixing the height
    img = img.resize((new_width, target_height), Image.LANCZOS)

    # Save the resized image
    img.save(output_path)

    return output_path

@spaces.GPU
def infer(use_custom_model, model_name, weight_name, custom_lora_weight, image_in, prompt, negative_prompt, preprocessor, controlnet_conditioning_scale, guidance_scale, inf_steps, seed, progress=gr.Progress(track_tqdm=True)):

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16, 
        variant="fp16",
        use_safetensors=True
    )
    
    pipe.to(device)
    
    prompt = prompt
    negative_prompt = negative_prompt

    if seed < 0 :
        seed = random.randint(0, 423538377342)
    
    generator = torch.Generator(device=device).manual_seed(seed)

    if image_in == None:
        raise gr.Error("You forgot to upload a source image.")
    
    image_in = resize_image(image_in, "resized_input.jpg", 1024)
    
    if preprocessor == "canny":

        image = load_image(image_in)

        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
    
    if use_custom_model:
        
        if model_name == "":
            raise gr.Error("you forgot to set a custom model name.")
        
        custom_model = model_name

        # This is where you load your trained weights
        if weight_name == "NO SAFETENSORS FILE": 
            pipe.load_lora_weights(
                custom_model,     
                low_cpu_mem_usage = True,
                use_auth_token = True
            )
    
        else:
            pipe.load_lora_weights(
                custom_model,
                weight_name = weight_name,        
                low_cpu_mem_usage = True,
                use_auth_token = True
            )
    
        lora_scale=custom_lora_weight

        images = pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            image=image, 
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            guidance_scale = float(guidance_scale),
            num_inference_steps=inf_steps,
            generator=generator,
            cross_attention_kwargs={"scale": lora_scale}
        ).images
    else:
        images = pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            image=image, 
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            guidance_scale = float(guidance_scale),
            num_inference_steps=inf_steps,
            generator=generator,
        ).images

    images[0].save(f"result.png")

    return f"result.png", seed

css="""
#col-container{
    margin: 0 auto;
    max-width: 720px;
    text-align: left;
}
div#warning-duplicate {
    background-color: #ebf5ff;
    padding: 0 10px 5px;
    margin: 20px 0;
}
div#warning-duplicate > .gr-prose > h2, div#warning-duplicate > .gr-prose > p {
    color: #0f4592!important;
}
div#warning-duplicate strong {
    color: #0f4592;
}
p.actions {
    display: flex;
    align-items: center;
    margin: 20px 0;
}
div#warning-duplicate .actions a {
    display: inline-block;
    margin-right: 10px;
}
button#load_model_btn{
    height: 46px;
}
#status_info{
    font-size: 0.9em;
}
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="gray",
).set(
    body_text_color_dark='*neutral_800',
    background_fill_primary_dark='*neutral_50',
    background_fill_secondary_dark='*neutral_50',
    border_color_accent_dark='*primary_300',
    border_color_primary_dark='*neutral_200',
    color_accent_soft_dark='*neutral_50',
    link_text_color_dark='*secondary_600',
    link_text_color_active_dark='*secondary_600',
    link_text_color_hover_dark='*secondary_700',
    link_text_color_visited_dark='*secondary_500',
    # code_background_fill_dark='*neutral_100',
    shadow_spread_dark='6px',
    block_background_fill_dark='white',
    block_label_background_fill_dark='*primary_100',
    block_label_text_color_dark='*primary_500',
    block_title_text_color_dark='*primary_500',
    checkbox_background_color_dark='*background_fill_primary',
    checkbox_background_color_selected_dark='*primary_600',
    checkbox_border_color_dark='*neutral_100',
    checkbox_border_color_focus_dark='*primary_500',
    checkbox_border_color_hover_dark='*neutral_300',
    checkbox_border_color_selected_dark='*primary_600',
    checkbox_label_background_fill_selected_dark='*primary_500',
    checkbox_label_text_color_selected_dark='white',
    error_background_fill_dark='#fef2f2',
    error_border_color_dark='#b91c1c',
    error_text_color_dark='#b91c1c',
    error_icon_color_dark='#b91c1c',
    input_background_fill_dark='white',
    input_background_fill_focus_dark='*secondary_500',
    input_border_color_dark='*neutral_50',
    input_border_color_focus_dark='*secondary_300',
    input_placeholder_color_dark='*neutral_400',
    slider_color_dark='*primary_500',
    stat_background_fill_dark='*primary_300',
    table_border_color_dark='*neutral_300',
    table_even_background_fill_dark='white',
    table_odd_background_fill_dark='*neutral_50',
    button_primary_background_fill_dark='*primary_500',
    button_primary_background_fill_hover_dark='*primary_400',
    button_primary_border_color_dark='*primary_00',
    button_secondary_background_fill_dark='whiite',
    button_secondary_background_fill_hover_dark='*neutral_100',
    button_secondary_border_color_dark='*neutral_200',
    button_secondary_text_color_dark='*neutral_800'
)


def create_inference_demo() -> gr.Blocks:
  
  with gr.Blocks(theme=theme, css=css) as demo:
      with gr.Column(elem_id="col-container"):
          if is_shared_ui:
              top_description = gr.HTML(f'''
                  <div class="gr-prose">
                      <h2><svg xmlns="http://www.w3.org/2000/svg" width="18px" height="18px" style="margin-right: 0px;display: inline-block;"fill="none"><path fill="#fff" d="M7 13.2a6.3 6.3 0 0 0 4.4-10.7A6.3 6.3 0 0 0 .6 6.9 6.3 6.3 0 0 0 7 13.2Z"/><path fill="#fff" fill-rule="evenodd" d="M7 0a6.9 6.9 0 0 1 4.8 11.8A6.9 6.9 0 0 1 0 7 6.9 6.9 0 0 1 7 0Zm0 0v.7V0ZM0 7h.6H0Zm7 6.8v-.6.6ZM13.7 7h-.6.6ZM9.1 1.7c-.7-.3-1.4-.4-2.2-.4a5.6 5.6 0 0 0-4 1.6 5.6 5.6 0 0 0-1.6 4 5.6 5.6 0 0 0 1.6 4 5.6 5.6 0 0 0 4 1.7 5.6 5.6 0 0 0 4-1.7 5.6 5.6 0 0 0 1.7-4 5.6 5.6 0 0 0-1.7-4c-.5-.5-1.1-.9-1.8-1.2Z" clip-rule="evenodd"/><path fill="#000" fill-rule="evenodd" d="M7 2.9a.8.8 0 1 1 0 1.5A.8.8 0 0 1 7 3ZM5.8 5.7c0-.4.3-.6.6-.6h.7c.3 0 .6.2.6.6v3.7h.5a.6.6 0 0 1 0 1.3H6a.6.6 0 0 1 0-1.3h.4v-3a.6.6 0 0 1-.6-.7Z" clip-rule="evenodd"/></svg>
                      Note: you might want to use a <strong>private</strong> custom LoRa model</h2>
                  </div>
              ''', elem_id="warning-duplicate")
          gr.HTML("""
  <h2 style="text-align: center;">Image Generation</h2>
  <p style="text-align: center;">SD-XL Control LoRas. Use StableDiffusion XL with <a href="https://huggingface.co/collections/diffusers/sdxl-controlnets-64f9c35846f3f06f5abe351f">Diffusers' SDXL ControlNets</a></p>

          """)

          use_custom_model = gr.Checkbox(visible=False, label="Use a custom pre-trained LoRa model ? (optional)", value=False, info="To use a private model, you'll need to duplicate the space with your own access token.")
          
          with gr.Blocks(visible=False) as custom_model_box:
              with gr.Row():
                  with gr.Column():
                      if not is_shared_ui:
                          your_username = api.whoami()["name"]
                          my_models = api.list_models(author=your_username, filter=["diffusers", "stable-diffusion-xl", 'lora'])
                          model_names = [item.modelId for item in my_models]
      
                      if not is_shared_ui:
                          custom_model = gr.Dropdown(
                              label = "Your custom model ID",
                              info="You can pick one of your private models",
                              choices = model_names,
                              allow_custom_value = True
                              #placeholder = "username/model_id"
                          )
                      else:
                          custom_model = gr.Textbox(
                              label="Your custom model ID", 
                              placeholder="your_username/your_trained_model_name", 
                              info="Make sure your model is set to PUBLIC"
                          )
                      
                      weight_name = gr.Dropdown(
                          label="Safetensors file", 
                          #value="pytorch_lora_weights.safetensors", 
                          info="specify which one if model has several .safetensors files",
                          allow_custom_value=True,
                          visible = False
                      )
                  with gr.Column():
                      with gr.Group():
                          load_model_btn = gr.Button("Load my model", elem_id="load_model_btn")
                          previous_model = gr.Textbox(
                              visible = False
                          )
                          model_status = gr.Textbox(
                              label = "model status",
                              show_label = False,
                              elem_id = "status_info"
                          )
                      trigger_word = gr.Textbox(label="Trigger word", interactive=False, visible=False)
          
          image_in = gr.Image(sources="upload", type="filepath")
          
          with gr.Row():
              
              with gr.Column():
                  with gr.Group():
                      prompt = gr.Textbox(label="Prompt")
                      negative_prompt = gr.Textbox(label="Negative prompt", value="extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured")
                  with gr.Group():
                      guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=8.8)
                      inf_steps = gr.Slider(label="Inference Steps", minimum="25", maximum="50", step=1, value=25)
                      custom_lora_weight = gr.Slider(label="Custom model weights", minimum=0.1, maximum=0.9, step=0.1, value=0.7)

              with gr.Column():
                  with gr.Group():
                      preprocessor = gr.Dropdown(label="Preprocessor", choices=["canny"], value="canny", interactive=False, info="For the moment, only canny is available")  
                      controlnet_conditioning_scale = gr.Slider(label="Controlnet conditioning Scale", minimum=0.1, maximum=0.9, step=0.01, value=0.3)
                  with gr.Group():
                      seed = gr.Slider(
                          label="Seed",
                          info = "-1 denotes a random seed",
                          minimum=-1,
                          maximum=423538377342,
                          step=1,
                          value=-1
                      )
                      last_used_seed = gr.Number(
                          label = "Last used seed",
                          info = "the seed used in the last generation",
                      )
          
          
          submit_btn = gr.Button("Submit")
          
          result = gr.Image(label="Result")
      
      use_custom_model.change(
          fn = check_use_custom_or_no,
          inputs =[use_custom_model],
          outputs = [custom_model_box],
          queue = False
      )
      custom_model.blur(
          fn=custom_model_changed,
          inputs = [custom_model, previous_model],
          outputs = [model_status],
          queue = False
      )
      load_model_btn.click(
          fn = load_model,
          inputs=[custom_model],
          outputs = [previous_model, model_status, weight_name, trigger_word],
          queue = False
      )
      submit_btn.click(
          fn = infer,
          inputs = [use_custom_model, custom_model, weight_name, custom_lora_weight, image_in, prompt, negative_prompt, preprocessor, controlnet_conditioning_scale, guidance_scale, inf_steps, seed],
          outputs = [result, last_used_seed]
      )

  return demo 
  

#demo.queue(max_size=12).launch(share=True)
