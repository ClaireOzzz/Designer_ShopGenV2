import gradio as gr
import torch
import os
import shutil
import requests
import subprocess
from subprocess import getoutput
from huggingface_hub import login, HfFileSystem, snapshot_download, HfApi, create_repo

from app_train import create_training_demo
from sdxl.app_inference import create_inference_demo

css="""
#col-container {max-width: 780px; margin-left: auto; margin-right: auto;}
#upl-dataset-group {background-color: none!important;}

div#warning-ready {
    background-color: #ecfdf5;
    padding: 0 10px 5px;
    margin: 20px 0;
}
div#warning-ready > .gr-prose > h2, div#warning-ready > .gr-prose > p {
    color: #057857!important;
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

div#warning-setgpu {
    background-color: #fff4eb;
    padding: 0 10px 5px;
    margin: 20px 0;
}

div#warning-setgpu > .gr-prose > h2, div#warning-setgpu > .gr-prose > p {
    color: #92220f!important;
}

div#warning-setgpu a, div#warning-setgpu b {
    color: #91230f;
}

div#warning-setgpu p.actions > a {
    display: inline-block;
    background: #1f1f23;
    border-radius: 40px;
    padding: 6px 24px;
    color: antiquewhite;
    text-decoration: none;
    font-weight: 600;
    font-size: 1.2em;
}

button#load-dataset-btn{
min-height: 60px;
}
"""

with gr.Blocks(css=css) as demo:

    gr.Markdown("SUTD x SUNS Shop Design Generator")
    with gr.Tab("Training"):
         create_training_demo()
    with gr.Tab("Generation"):
        create_inference_demo()
    with gr.Tab("Visualisation"):
        gr.Markdown('''
            - You can use this tab to upload models later if you choose not to upload models in training time or if upload in training time failed.
            ''')


demo.queue(max_size=1).launch(debug=True, share=True)