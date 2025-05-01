# This repo is only SECourses followers Download Link : https://www.patreon.com/posts/127855145

https://github.com/user-attachments/assets/3d58b64d-2f99-4ab3-9a81-200839a0869e

![screencapture-127-0-0-1-7860-2025-05-01-13_12_32](https://github.com/user-attachments/assets/69044597-29bc-49e2-b677-a544607c01c4)


FantasyTalking SECourses App Extra Features
By monitoring your VRAM you can increase Persistent Params Value - we already have presets

Higher Resolution uses more VRAM

More duration uses more VRAM - sadly

I have uploaded model files to my repo for faster new XET download (Hugging Face gave me XET feature) - no errors and fast :D

Application supports RTX 5000 series and below GPUs, auto installs Torch 2.7 with CUDA 12.8, Flash Attention, DeepSpeed, Sage Attention, xFormers, Triton

I have completely rewritten the demo Gradio on the repo, added so many new features. Some of them as below:

Hard coded parameters are fixed and now you can set each one on Gradio

You can set different width and height

You can set different steps count

You can set different FPS - directly impacts how many frames generated

You can set Output Quality (CRF)

All generation metadata will be saved along with the videos

Prompt and negative prompt box with set-able CFG values

Cancel generation feature implemented - had to modify pipeline files as well

The models will be downloaded inside models sub folder

I started using single FP16 model file because I will hopefully update Wan 2.1 app to use same file as well so you can symlink same file

Make sure to set at least 50 GB virtual RAM or more

Every generation and used audio file will be saved inside outputs folder and used_audios folder

Batch processing fully implemented logic is easy

image_filename.png, image_filename.txt, image_filename.mp3

Supports ".wav", ".mp3", ".flac" and ".png", ".jpg", ".jpeg", ".webp", ".bmp"

Upload video feature implemented

When you upload a video it will auto extract its audio and put it into audio input place

You can use Gradio interface audio trim feature to get exact audio duration

The progress will be both shown on Gradio interface and also on CMD window

First frame of the generated video will be input image - higher quality

Default prompt set to A person is talking and making hand gestures.
