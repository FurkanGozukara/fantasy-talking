import argparse
import sys
from datetime import datetime
from pathlib import Path
import math
import random

import gradio as gr
import librosa
import torch

from infer import load_models, main

pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None
current_torch_dtype = None
current_num_persistent_param_in_dit = None
models_loaded = False

DEFAULT_PROMPT = "A person is talking."
DEFAULT_NEGATIVE_PROMPT = "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_FPS = 23
DEFAULT_DURATION = 5
MAX_DURATION = 60
DEFAULT_SEED = 1247
DEFAULT_INFERENCE_STEPS = 20
DEFAULT_PROMPT_CFG = 5.0
DEFAULT_AUDIO_CFG = 5.0
DEFAULT_AUDIO_WEIGHT = 1.0
DEFAULT_SIGMA_SHIFT = 5.0
DEFAULT_DENOISING_STRENGTH = 1.0
DEFAULT_SAVE_QUALITY = 10
DEFAULT_TILE_SIZE_H = 30
DEFAULT_TILE_SIZE_W = 52
DEFAULT_TILE_STRIDE_H = 15
DEFAULT_TILE_STRIDE_W = 26

MODEL_DIRS = {
    "wan_model_dir": "./models",
    "fantasytalking_model_path": "./models/fantasytalking_model.ckpt",
    "wav2vec_model_dir": "./models/wav2vec2-base-960h",
}

VRAM_PRESETS = {
    "6GB GPUs": "0",
    "8GB GPUs": "0",
    "10GB GPUs": "0",
    "12GB GPUs": "0",
    "16GB GPUs": "0",
    "24GB GPUs": "6,000,000,000",
    "32GB GPUs": "14,000,000,000",
    "48GB GPUs": "22,000,000,000",
    "80GB GPUs": "30,000,000,000",
}
VRAM_PRESET_DEFAULT = "24GB GPUs"

TORCH_DTYPES_STR = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
TORCH_DTYPE_DEFAULT = "bfloat16"


def calculate_frames(duration_sec, fps):
    total_frames = math.ceil(duration_sec * fps)
    k = math.ceil((total_frames - 1) / 4)
    num_frames = 4 * k + 1
    return num_frames

def get_persistent_params(preset_name):
    return VRAM_PRESETS.get(preset_name, VRAM_PRESETS[VRAM_PRESET_DEFAULT])

def get_torch_dtype(dtype_str):
    return TORCH_DTYPES_STR.get(dtype_str, TORCH_DTYPES_STR[TORCH_DTYPE_DEFAULT])

def generate_video(
    image_path,
    audio_path,
    prompt,
    negative_prompt,
    width,
    height,
    duration_seconds,
    fps,
    prompt_cfg_scale,
    audio_cfg_scale,
    audio_weight,
    inference_steps,
    seed,
    use_random_seed,
    tiled_vae,
    tile_size_h,
    tile_size_w,
    tile_stride_h,
    tile_stride_w,
    vram_preset_name,
    torch_dtype_str,
    sigma_shift,
    denoising_strength,
    save_video_quality,
    progress=gr.Progress(track_tqdm=True)
):
    # --- Declare globals at the beginning --- #
    global pipe, fantasytalking, wav2vec_processor, wav2vec, models_loaded
    global current_torch_dtype, current_num_persistent_param_in_dit

    # --- Input Validation ---
    if image_path is None:
        raise gr.Error("Input Image is required. Please upload an image.")
    if audio_path is None:
        raise gr.Error("Input Audio is required. Please upload or record audio.")

    progress(0, desc="Preparing parameters...")

    if not isinstance(duration_seconds, (int, float)) or duration_seconds <= 0:
        duration_seconds = DEFAULT_DURATION
        gr.Warning(f"Invalid duration, using default: {DEFAULT_DURATION}s")
    duration_seconds = min(duration_seconds, MAX_DURATION)

    if not isinstance(width, int) or width <= 0 or width % 16 != 0:
        width = DEFAULT_WIDTH
        gr.Warning(f"Invalid width or not divisible by 16, using default: {DEFAULT_WIDTH}")
    if not isinstance(height, int) or height <= 0 or height % 16 != 0:
        height = DEFAULT_HEIGHT
        gr.Warning(f"Invalid height or not divisible by 16, using default: {DEFAULT_HEIGHT}")

    if use_random_seed or seed is None or not isinstance(seed, int) or seed < 0:
        seed = random.randint(0, 2**32 - 1)
        print(f"Using random seed: {seed}")
    else:
        seed = int(seed)
        print(f"Using fixed seed: {seed}")

    try:
        actual_audio_duration = librosa.get_duration(filename=audio_path)
        print(f"Actual audio duration: {actual_audio_duration:.2f}s")
        target_duration = min(duration_seconds, actual_audio_duration)
        if target_duration < duration_seconds:
            gr.Warning(f"Requested duration ({duration_seconds}s) is longer than audio ({actual_audio_duration:.2f}s). Using audio duration.")

        num_frames = calculate_frames(target_duration, fps)
        print(f"Target duration: {target_duration:.2f}s, Calculated num_frames: {num_frames}")

    except Exception as e:
        print(f"Error getting audio duration: {e}")
        raise gr.Error(f"Could not read audio file: {audio_path}. Error: {e}")

    num_persistent_param_in_dit = get_persistent_params(vram_preset_name)
    torch_dtype = get_torch_dtype(torch_dtype_str)

    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path).resolve().as_posix()
    audio_path = Path(audio_path).resolve().as_posix()

    args_dict = {
        "image_path": image_path,
        "audio_path": audio_path,
        "prompt": prompt if prompt else DEFAULT_PROMPT,
        "negative_prompt": negative_prompt,
        "output_dir": str(output_dir),
        "width": int(width),
        "height": int(height),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "audio_weight": float(audio_weight),
        "prompt_cfg_scale": float(prompt_cfg_scale),
        "audio_cfg_scale": float(audio_cfg_scale),
        "inference_steps": int(inference_steps),
        "seed": int(seed),
        "tiled_vae": bool(tiled_vae),
        "tile_size_h": int(tile_size_h),
        "tile_size_w": int(tile_size_w),
        "tile_stride_h": int(tile_stride_h),
        "tile_stride_w": int(tile_stride_w),
        "sigma_shift": float(sigma_shift),
        "denoising_strength": float(denoising_strength),
        "save_video_quality": int(save_video_quality),
    }


    load_needed = False
    if not models_loaded:
        load_needed = True
        print("Models not loaded yet.")
    elif current_torch_dtype != torch_dtype:
        load_needed = True
        print(f"Torch dtype changed ({current_torch_dtype} -> {torch_dtype}). Reloading models.")
    elif current_num_persistent_param_in_dit != num_persistent_param_in_dit:
        load_needed = True
        print(f"VRAM persistence changed ({current_num_persistent_param_in_dit} -> {num_persistent_param_in_dit}). Reloading models.")

    if load_needed:
        progress(0.1, desc="Loading models (may take a while)...")
        print("Unloading previous models if any...")
        pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None
        torch.cuda.empty_cache()
        try:
            pipe, fantasytalking, wav2vec_processor, wav2vec = load_models(
                wan_model_dir=MODEL_DIRS["wan_model_dir"],
                fantasytalking_model_path=MODEL_DIRS["fantasytalking_model_path"],
                wav2vec_model_dir=MODEL_DIRS["wav2vec_model_dir"],
                num_persistent_param_in_dit=num_persistent_param_in_dit,
                torch_dtype=torch_dtype,
                device="cuda"
            )
            models_loaded = True
            current_torch_dtype = torch_dtype
            current_num_persistent_param_in_dit = num_persistent_param_in_dit
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            models_loaded = False
            pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None
            torch.cuda.empty_cache()
            raise gr.Error(f"Failed to load models. Check paths and VRAM. Error: {str(e)}")
    else:
        print("Models already loaded and configuration matches. Skipping reload.")

    progress(0.3, desc="Starting video generation...")
    print("Calling infer.main...")
    output_path = None # Initialize output_path
    try:
        output_path = main(args_dict, pipe, fantasytalking, wav2vec_processor, wav2vec)
        progress(1.0, desc="Generation complete!")
        print(f"Video generated successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"An error occurred during 'infer.main': {str(e)}")
        print(f"Exception type: {type(e)}") # Log the exception type for debugging
        print("Attempting to clear CUDA cache...")
        torch.cuda.empty_cache()
        print("CUDA cache clear attempted.")
        # Raise a Gradio error to notify the user
        if "interrupted" in str(e).lower(): # Basic check for interruption message
             raise gr.Error(f"Generation cancelled by user.")
        else:
             raise gr.Error(f"Error during video generation: {str(e)}")

with gr.Blocks(title="FantasyTalking Video Generation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # FantasyTalking: Realistic Talking Portrait Generation SECourses App V1 - https://www.patreon.com/posts/127855145
    Generate a talking head video from an image and audio.
    Configure various settings below to control the output.
    [GitHub](https://github.com/Fantasy-AMAP/fantasy-talking) | [arXiv Paper](https://arxiv.org/abs/2504.04842)
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Inputs")
            image_input = gr.Image(label="Input Image", type="filepath")
            audio_input = gr.Audio(label="Input Audio (WAV recommended)", type="filepath")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder=DEFAULT_PROMPT,
                value=DEFAULT_PROMPT,
                lines=3,
                info="Describe the scene or action (e.g., 'A woman is talking.')"
            )
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                placeholder=DEFAULT_NEGATIVE_PROMPT,
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=3,
                info="Describe what NOT to generate. Active when CFG Scale > 1.0"
            )

            gr.Examples(
                examples=[
                    [
                        "assets/images/woman.png",
                        "assets/audios/woman.wav",
                        "A woman is talking."
                    ]
                ],
                inputs=[image_input, audio_input, prompt_input],
                label="Examples (Image, Audio, Prompt)"
            )

        with gr.Column(scale=1):
            with gr.Row():
                process_btn = gr.Button("Generate Video", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="stop")                
            gr.Markdown("## 2. Generation Settings")
            with gr.Row():
                 width_input = gr.Number(value=DEFAULT_WIDTH, label="Width", precision=0, info="Output video width (must be divisible by 16)")
                 height_input = gr.Number(value=DEFAULT_HEIGHT, label="Height", precision=0, info="Output video height (must be divisible by 16)")
            with gr.Row():
                duration_input = gr.Number(value=DEFAULT_DURATION, minimum=1, maximum=MAX_DURATION, label="Max Duration (s)", info=f"Target video length in seconds (capped by audio length, max {MAX_DURATION}s)")
                fps_input = gr.Number(value=DEFAULT_FPS, minimum=1, maximum=60, label="FPS", precision=0, info="Output video frames per second")
            gr.Markdown("_Video length affects processing time. Total frames = `ceil(duration * fps)`, adjusted to `4k+1` format._")

            with gr.Row():
                prompt_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, value=DEFAULT_PROMPT_CFG, step=0.5, label="Prompt CFG Scale", info="How much to follow the text prompt.")
                audio_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, value=DEFAULT_AUDIO_CFG, step=0.5, label="Audio CFG Scale", info="How much to follow the audio lip sync.")
            audio_weight = gr.Slider(minimum=0.1, maximum=3.0, value=DEFAULT_AUDIO_WEIGHT, step=0.1, label="Audio Weight", info="Overall strength of audio influence.")
            inference_steps = gr.Slider(minimum=1, maximum=100, value=DEFAULT_INFERENCE_STEPS, step=1, label="Inference Steps", info="More steps = potentially better quality, longer time.")

            with gr.Row():
                seed_input = gr.Number(value=DEFAULT_SEED, label="Seed", precision=0)
                random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=True)

            with gr.Accordion("Advanced Settings", open=True):
                with gr.Row():
                    sigma_shift = gr.Slider(minimum=0.1, maximum=10.0, value=DEFAULT_SIGMA_SHIFT, step=0.1, label="Sigma Shift (Scheduler)", info="FlowMatchScheduler parameter.")
                    denoising_strength = gr.Slider(minimum=0.1, maximum=1.0, value=DEFAULT_DENOISING_STRENGTH, step=0.05, label="Denoising Strength (Scheduler)", info="Scheduler parameter (usually 1.0 for I2V).")
                save_video_quality = gr.Slider(minimum=0, maximum=10, value=DEFAULT_SAVE_QUALITY, step=1, label="Output Video Quality (ffmpeg)", info="Quality for saving final video (0-10, 10=best).")

            with gr.Accordion("Performance & VRAM", open=True):
                gr.Markdown("_Higher resolution and longer duration require more VRAM._")
                vram_preset_dropdown = gr.Dropdown(
                    choices=list(VRAM_PRESETS.keys()),
                    value=VRAM_PRESET_DEFAULT,
                    label="VRAM Usage Preset",
                    info="Adjusts how many model parts stay in VRAM. 'Minimum' is slowest but uses least VRAM."
                )
                torch_dtype_dropdown = gr.Dropdown(
                    choices=list(TORCH_DTYPES_STR.keys()),
                    value=TORCH_DTYPE_DEFAULT,
                    label="Computation Precision (torch_dtype)",
                    info="Affects speed and VRAM. 'bfloat16' recommended if supported. Requires model reload."
                )
                tiled_vae_checkbox = gr.Checkbox(label="Enable Tiled VAE", value=True, info="Use less VRAM during VAE encoding/decoding, may be slightly slower.")
                with gr.Group(visible=True) as tile_options:
                    gr.Markdown("**Tiling Options (if Tiled VAE enabled):**")
                    with gr.Row():
                        tile_size_h_input = gr.Number(value=DEFAULT_TILE_SIZE_H, label="Tile Height", precision=0, info="Tile size in latent space.")
                        tile_size_w_input = gr.Number(value=DEFAULT_TILE_SIZE_W, label="Tile Width", precision=0)
                    with gr.Row():
                        tile_stride_h_input = gr.Number(value=DEFAULT_TILE_STRIDE_H, label="Tile Stride H", precision=0, info="Overlap between tiles.")
                        tile_stride_w_input = gr.Number(value=DEFAULT_TILE_STRIDE_W, label="Tile Stride W", precision=0)



        with gr.Column(scale=1):
            gr.Markdown("## 3. Output")
            video_output = gr.Video(label="Generated Video")

    gen_event = process_btn.click(
        fn=generate_video,
        inputs=[
            image_input,
            audio_input,
            prompt_input,
            negative_prompt_input,
            width_input,
            height_input,
            duration_input,
            fps_input,
            prompt_cfg_scale,
            audio_cfg_scale,
            audio_weight,
            inference_steps,
            seed_input,
            random_seed_checkbox,
            tiled_vae_checkbox,
            tile_size_h_input,
            tile_size_w_input,
            tile_stride_h_input,
            tile_stride_w_input,
            vram_preset_dropdown,
            torch_dtype_dropdown,
            sigma_shift,
            denoising_strength,
            save_video_quality,
        ],
        outputs=video_output,
    )
    cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    args, unknown = parser.parse_known_args()

    share_flag = args.share

    print(f"Launching Gradio app... Share={share_flag}")
    demo.launch(inbrowser=True, share=share_flag)
