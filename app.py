import gradio as gr
import torch
import os
import shutil
import requests
import subprocess
from subprocess import getoutput
from huggingface_hub import login, HfFileSystem, snapshot_download, HfApi, create_repo
from pathlib import Path
from PIL import Image

from app_train import create_training_demo
from sdxl.app_inference import create_inference_demo
from depthgltf.app_visualisations import create_visual_demo 
from inpaint.app_inpaint import create_inpaint_demo

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import numpy as np
import open3d as o3d


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

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("# SUTD x SUNS Shop Design Generator for Designers")
    gr.Markdown("Train, Use and Visualise in 2.5D")
    with gr.Tab("Training"):
         create_training_demo()
    with gr.Tab("Generation"):
        create_inference_demo()
    with gr.Tab("Edit Image"):
      create_inpaint_demo()
    with gr.Tab("Visualisation"):
        create_visual_demo(); 
demo.queue().launch(debug=True, share=True)

