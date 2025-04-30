# --- START OF FILE infer.py ---

# Copyright Alibaba Inc. All Rights Reserved.

import os
import subprocess
from datetime import datetime
from pathlib import Path
import time # For calculating duration
import json # For metadata saving
import pprint # For pretty printing dicts
import traceback # For detailed error logging

import cv2
# import librosa # No longer needed here, handled in gradio app
import torch
from PIL import Image
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np # Keep for potential future use (e.g. seed generation if moved here)

from diffsynth import ModelManager, WanVideoPipeline
from model import FantasyTalkingAudioConditionModel
from utils import get_audio_features, resize_image_by_longest_edge, save_video

# --- Import the custom exception --- #
# Ensure wan_video.py is accessible or the CancelledError is defined here
try:
    # Use the specific exception from the pipeline module if available
    from diffsynth.pipelines.wan_video import CancelledError
except ImportError:
    print("Warning: Could not import CancelledError from diffsynth.pipelines.wan_video. Defining locally.")
    class CancelledError(Exception):
        """Custom exception for cancellation."""
        pass
# ---------------------------------- #


def load_models(
    wan_model_dir: str,
    fantasytalking_model_path: str,
    wav2vec_model_dir: str,
    num_persistent_param_in_dit: int | None, # Accept int or None
    torch_dtype: torch.dtype,
    device: str = "cuda",
):
    """Loads the required models."""
    # --- Enhanced Logging ---
    print("-" * 20)
    print("Attempting to load models...")
    print(f"Target Device: {device}")
    print(f"Target Dtype: {torch_dtype}")
    print(f"Wan Model Dir: {wan_model_dir}")
    print(f"FantasyTalking Model Path: {fantasytalking_model_path}")
    print(f"Wav2Vec Model Dir: {wav2vec_model_dir}")
    # Handle None value for printing
    persistent_params_str = str(num_persistent_param_in_dit) if num_persistent_param_in_dit is not None else "None (Unlimited)"
    print(f"Persistent Params (parsed value): {persistent_params_str}")
    # ------------------------

    # Load Wan I2V models
    print("Loading Wan I2V models to CPU first...") # Log added
    model_manager = ModelManager(device="cpu") # Keep on CPU initially
    model_manager.load_models(
        [
            f"{wan_model_dir}/wan21_i2v_720p_14B_fp16.safetensors",
            f"{wan_model_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            f"{wan_model_dir}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{wan_model_dir}/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch_dtype, # Load with target dtype even on CPU if possible
    )
    print("Wan I2V models loaded to CPU.") # Log added
    print("Creating WanVideoPipeline...") # Log added
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch_dtype, device=device # Pipeline will manage moving to target device
    )
    print("WanVideoPipeline created.") # Log added

    # Load FantasyTalking weights
    print("Loading FantasyTalking weights...") # Log added
    fantasytalking = FantasyTalkingAudioConditionModel(pipe.dit, 768, 2048).to(device)
    fantasytalking.load_audio_processor(fantasytalking_model_path, pipe.dit)
    print("FantasyTalking weights loaded and processor attached.") # Log added

    # Enable VRAM management
    # --- Updated Logic for None/int ---
    if num_persistent_param_in_dit is not None:
         print(f"Enabling VRAM management with num_persistent_param_in_dit: {num_persistent_param_in_dit}")
         # Value should already be int or None from Gradio app parsing
         pipe.enable_vram_management(
             num_persistent_param_in_dit=num_persistent_param_in_dit # Pass int directly
         )
    else:
        print("VRAM management disabled (all parameters persistent - 'None' value used).")
    # ---------------------------------

    # Load wav2vec models
    print("Loading Wav2Vec models...") # Log added
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
    wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_dir).to(device)
    print("Wav2Vec models loaded.") # Log added
    print("-" * 20) # Log added
    print("Models loaded successfully.") # Log added
    return pipe, fantasytalking, wav2vec_processor, wav2vec


# Add cancel_fn parameter
def main(args: dict, pipe: WanVideoPipeline, fantasytalking: FantasyTalkingAudioConditionModel, wav2vec_processor: Wav2Vec2Processor, wav2vec: Wav2Vec2Model, cancel_fn=None):
    """Generates the video based on the provided arguments and loaded models."""
    start_time = datetime.now() # Add start time for duration calculation
    start_time_perf = time.perf_counter() # More precise timer

    # Get generation index/total for logging
    generation_index = args.get('generation_index', 1)
    total_generations = args.get('total_generations', 1) # If not passed, assume 1
    log_prefix = f"[Gen {generation_index}/{total_generations}]" # Prefix for logs

    # --- Enhanced Logging ---
    print(f"\n{log_prefix} ===== Starting Generation =====")
    print(f"{log_prefix} Received args:")
    # Use pprint for better readability, exclude potentially large tensors if any sneak in
    printable_args = {k: v if not isinstance(v, (torch.Tensor, np.ndarray)) else f"{type(v).__name__} shape {getattr(v, 'shape', 'N/A')}" for k, v in args.items()}
    pprint.pprint(printable_args, indent=2)
    # ------------------------

    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"{log_prefix} Ensured output directory exists: {output_dir}") # Log added

    # --- Unique Filenaming Logic (Handles Batch and Sequential) ---
    output_base_name_arg = args.get('output_base_name') # For batch mode override
    save_metadata = args.get('save_metadata', False) # Check if metadata saving is enabled

    if output_base_name_arg:
        # Batch mode: Use the provided stem directly
        base_name = output_base_name_arg
        print(f"{log_prefix} Using provided base name for batch mode: {base_name}")
        save_path = os.path.join(output_dir, f"{base_name}.mp4")
        metadata_path = os.path.join(output_dir, f"{base_name}.txt")
        # Check for existing file (Gradio app handles skipping, this is a safety log)
        if os.path.exists(save_path):
             print(f"{log_prefix} Warning: Output file {save_path} already exists (batch mode). It will be overwritten by ffmpeg.") # Note: ffmpeg -y handles overwrite
    else:
        # Single/Sequential mode: Find the next available 000X number
        print(f"{log_prefix} Determining sequential output filename (e.g., 0001.mp4)...")
        sequence_num = 1
        while True:
            base_name = f"{sequence_num:04d}"
            save_path = os.path.join(output_dir, f"{base_name}.mp4")
            metadata_path = os.path.join(output_dir, f"{base_name}.txt")
            # Check if EITHER the video OR the metadata file exists to reserve the number
            video_exists = os.path.exists(save_path)
            metadata_exists = save_metadata and os.path.exists(metadata_path)

            if not video_exists and not metadata_exists:
                print(f"{log_prefix} Using unique sequential base name: {base_name}")
                break

            # Log why we are skipping this number
            if video_exists:
                print(f"{log_prefix} Skipping sequence {sequence_num}: Video file '{base_name}.mp4' already exists.")
            if metadata_exists:
                 print(f"{log_prefix} Skipping sequence {sequence_num}: Metadata file '{base_name}.txt' already exists.")

            sequence_num += 1
            if sequence_num > 99999: # Increased safety break limit
                raise RuntimeError(f"{log_prefix} Could not find a unique filename after {sequence_num-1} attempts in {output_dir}. Please clean the directory.")
    # --- End Filenaming Logic ---

    # Define temporary path using the determined base_name
    save_path_tmp = os.path.join(output_dir, f"tmp_{base_name}_{int(time.time())}.mp4") # Add timestamp for more uniqueness

    # --- Enhanced Logging ---
    print(f"{log_prefix} Final Save Path: {save_path}")
    if save_metadata:
        print(f"{log_prefix} Metadata Save Path: {metadata_path}")
    print(f"{log_prefix} Temporary Video Path: {save_path_tmp}")
    # ------------------------


    # --- Parameter Extraction ---
    num_frames = args['num_frames']
    fps = args['fps']
    audio_path = args['audio_path']
    image_path = args['image_path']
    width = args['width']
    height = args['height']
    seed = args['seed'] # Use the seed passed (already handled random/increment in Gradio)

    # --- Enhanced Logging ---
    print(f"{log_prefix} Processing Image: {Path(image_path).name}")
    print(f"{log_prefix} Processing Audio: {Path(audio_path).name}")
    print(f"{log_prefix} Target dimensions: {width}x{height}, Frames: {num_frames}, FPS: {fps}, Seed: {seed}")
    # ------------------------

    # --- Load and Resize Image ---
    try:
        print(f"{log_prefix} Loading and resizing image from: {image_path} to {width}x{height}") # Log added
        image = Image.open(image_path).convert("RGB")
        # Use LANCZOS for high-quality resize
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        print(f"{log_prefix} Resized image size: {image.size}") # Log added
    except Exception as e:
        # --- Enhanced Error Handling ---
        print(f"{log_prefix} !!! Error loading or resizing image: {e}")
        traceback.print_exc()
        raise # Re-raise critical error
        # -----------------------------


    # --- Extract Audio Features ---
    try:
        print(f"{log_prefix} Extracting audio features (Target FPS={fps}, Target NumFrames={num_frames})...") # Log added
        audio_wav2vec_fea = get_audio_features(
            wav2vec, wav2vec_processor, audio_path, fps, num_frames, device=device # Pass device
        )
        print(f"{log_prefix} Audio Wav2Vec features extracted, shape: {audio_wav2vec_fea.shape}, device: {audio_wav2vec_fea.device}") # Log added
    except Exception as e:
        # --- Enhanced Error Handling ---
        print(f"{log_prefix} !!! Error extracting audio features: {e}")
        traceback.print_exc()
        raise # Re-raise critical error
        # -----------------------------

    # --- Process Audio Features ---
    try:
        print(f"{log_prefix} Projecting audio features...") # Log added
        audio_proj_fea = fantasytalking.get_proj_fea(audio_wav2vec_fea)
        print(f"{log_prefix} Audio projected features calculated, shape: {audio_proj_fea.shape}") # Log added
        pos_idx_ranges = fantasytalking.split_audio_sequence(
            audio_proj_fea.size(1), num_frames=num_frames
        )
        print(f"{log_prefix} Splitting audio features for pipeline...") # Log added
        audio_proj_split, audio_context_lens = fantasytalking.split_tensor_with_padding(
            audio_proj_fea, pos_idx_ranges, expand_length=4
        )
        print(f"{log_prefix} Audio features split, shape: {audio_proj_split.shape}, context lengths calculated.") # Log added
    except Exception as e:
        # --- Enhanced Error Handling ---
        print(f"{log_prefix} !!! Error processing projected audio features: {e}")
        traceback.print_exc()
        raise # Re-raise critical error
        # -----------------------------

    # --- Set sigma_shift (optional scheduler parameter) ---
    # This depends heavily on the specific scheduler implementation in WanVideoPipeline
    try:
        # Check if scheduler exists and has necessary attributes/methods
        if hasattr(pipe, 'scheduler') and pipe.scheduler is not None:
            if hasattr(pipe.scheduler, 'config') and isinstance(pipe.scheduler.config, dict) and 'sigma_shift' in pipe.scheduler.config:
                 print(f"{log_prefix} Setting scheduler sigma_shift via config: {args['sigma_shift']}")
                 pipe.scheduler.config['sigma_shift'] = args['sigma_shift']
                 # Some schedulers might need re-initialization or update after config change
                 if hasattr(pipe.scheduler, 'set_timesteps'):
                     # Example: Re-set timesteps if needed (adapt parameters as necessary)
                     # pipe.scheduler.set_timesteps(num_inference_steps=args['inference_steps'], denoising_strength=args['denoising_strength'])
                     pass # Currently assume direct config change is sufficient for FlowMatchScheduler used here
                 else:
                      print(f"{log_prefix} Scheduler lacks set_timesteps, direct config change applied.")

            elif hasattr(pipe.scheduler, 'shift'): # Alternative: if shift is a direct attribute
                 print(f"{log_prefix} Setting scheduler shift attribute directly: {args['sigma_shift']}")
                 pipe.scheduler.shift = args['sigma_shift']
                 if hasattr(pipe.scheduler, 'set_timesteps'):
                     # Example: Re-set timesteps if needed
                     # pipe.scheduler.set_timesteps(num_inference_steps=args['inference_steps'], denoising_strength=args['denoising_strength'])
                      pass
                 else:
                      print(f"{log_prefix} Scheduler lacks set_timesteps, direct attribute change applied.")
            else:
                 print(f"{log_prefix} Scheduler does not support direct sigma_shift configuration via 'config' dict or 'shift' attribute.")
        else:
             print(f"{log_prefix} Warning: Pipeline scheduler not found or not initialized. Cannot set sigma_shift.")
    except Exception as e:
        print(f"{log_prefix} Warning: Could not set sigma_shift on scheduler: {e}")


    # --- Prepare Diffusion Pipeline Arguments ---
    print(f"{log_prefix} Preparing arguments for WanVideoPipeline...") # Log added
    try:
        pipe_kwargs = {
            "prompt": args['prompt'],
            "negative_prompt": args['negative_prompt'],
            "input_image": image, # Use the resized PIL Image object
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": args['inference_steps'],
            "seed": seed,
            "tiled": args['tiled_vae'], # Use the boolean value directly
            "audio_scale": args['audio_weight'], # Map from Gradio 'audio_weight'
            "cfg_scale": args['prompt_cfg_scale'],
            "audio_cfg_scale": args['audio_cfg_scale'],
            "audio_proj": audio_proj_split.to(dtype=pipe.torch_dtype), # Ensure correct dtype
            "audio_context_lens": audio_context_lens,
            "latents_num_frames": (num_frames - 1) // 4 + 1,
            "denoising_strength": args['denoising_strength'], # Pass denoising strength
            # Cancel function will be passed separately below
        }
        if args['tiled_vae']:
            # Ensure tile sizes/strides are integers
            tile_size_h = int(args['tile_size_h'])
            tile_size_w = int(args['tile_size_w'])
            tile_stride_h = int(args['tile_stride_h'])
            tile_stride_w = int(args['tile_stride_w'])
            pipe_kwargs["tile_size"] = (tile_size_h, tile_size_w)
            pipe_kwargs["tile_stride"] = (tile_stride_h, tile_stride_w)
            print(f"{log_prefix} Tiling enabled with size: {pipe_kwargs['tile_size']}, stride: {pipe_kwargs['tile_stride']}") # Log added
        else:
            print(f"{log_prefix} Tiling disabled.") # Log added

        # --- Enhanced Logging ---
        print(f"{log_prefix} Pipeline arguments prepared (excluding large tensors):")
        # Create a printable version excluding actual tensors
        printable_pipe_kwargs = {}
        for k, v in pipe_kwargs.items():
            if isinstance(v, torch.Tensor):
                printable_pipe_kwargs[k] = f"Tensor shape {v.shape} on {v.device} dtype {v.dtype}"
            elif isinstance(v, Image.Image):
                 printable_pipe_kwargs[k] = f"PIL Image size {v.size} mode {v.mode}"
            else:
                 printable_pipe_kwargs[k] = v
        pprint.pprint(printable_pipe_kwargs, indent=2)
        # ------------------------

    except Exception as e:
        # --- Enhanced Error Handling ---
        print(f"{log_prefix} !!! Error preparing pipeline arguments: {e}")
        traceback.print_exc()
        raise # Re-raise critical error
        # -----------------------------


    # --- Image-to-Video Diffusion Pipeline Execution ---
    print(f"{log_prefix} Starting WanVideoPipeline generation...") # Log added
    video_frames_pil = None # Initialize variable
    try:
        # *** Pass cancel_fn to the pipeline call ***
        video_frames_pil = pipe(**pipe_kwargs, cancel_fn=cancel_fn) # Pass cancel_fn here
        # *******************************************
        print(f"{log_prefix} WanVideoPipeline finished successfully. Received {len(video_frames_pil)} PIL frames.") # Log added
    except CancelledError as e: # Catch the specific error from the pipeline
        print(f"{log_prefix} Pipeline cancelled during execution: {e}") # Log added
        # Attempt to unload models if VRAM management is active
        # Check if the unloading method exists and is callable
        if hasattr(pipe, 'load_models_to_device') and callable(getattr(pipe, 'load_models_to_device')):
            print(f"{log_prefix} Attempting to unload/offload models from GPU due to cancellation...")
            try:
                # Calling load_models_to_device with empty list triggers offloading in diffsynth's implementation
                pipe.load_models_to_device([])
                print(f"{log_prefix} Models unloaded/offloaded.") # Log added
            except Exception as unload_e:
                print(f"{log_prefix} Warning: Error during model unloading on cancel: {unload_e}") # Log added
        else:
             print(f"{log_prefix} Model unloading mechanism not found on pipeline object.") # Log added
        raise # Re-raise CancelledError to be caught by Gradio app loop/handler
    except Exception as e:
        # --- Enhanced Error Handling ---
        print(f"{log_prefix} !!! Critical Error during WanVideoPipeline execution: {e}")
        traceback.print_exc() # Print full traceback for debugging
        # Attempt to clear cache even on general errors
        print(f"{log_prefix} Attempting to clear CUDA cache after pipeline error...")
        torch.cuda.empty_cache()
        print(f"{log_prefix} CUDA cache clear attempted.")
        raise # Re-raise the original exception
        # -----------------------------


    # --- Save Temporary Video (using diffsynth's save_video) ---
    try:
        # Convert PIL images to numpy array (T, H, W, C) expected by save_video
        video_array = np.stack([np.array(frame) for frame in video_frames_pil])
        print(f"{log_prefix} Saving {video_array.shape[0]} generated frames to temporary video: {save_path_tmp}") # Log added
        # Note: save_video quality parameter might not directly map to CRF.
        # Use CRF in ffmpeg for final quality control.
        save_video(video_array, save_path_tmp, fps=fps) # Use default quality for temp file
        print(f"{log_prefix} Temporary video saved.") # Log added
    except Exception as e:
        # --- Enhanced Error Handling ---
        print(f"{log_prefix} !!! Error saving temporary video using save_video: {e}")
        traceback.print_exc()
        raise # Re-raise critical error
        # -----------------------------


    # --- Combine Video and Audio using FFmpeg ---
    # Use CRF for quality control
    crf_value = args.get('save_video_quality', 18) # Get CRF from args (adjust range if needed), default 18
    print(f"{log_prefix} Attempting to merge video and audio using ffmpeg (CRF: {crf_value})...") # Log added
    # Use a more robust ffmpeg command
    final_command = [
        "ffmpeg",
        "-y", # Overwrite output file without asking
        "-i", save_path_tmp, # Input temporary video
        "-i", audio_path, # Input original audio
        "-map", "0:v:0", # Map video stream from first input
        "-map", "1:a:0?", # Map audio stream from second input, '?' makes it optional if audio fails/is absent
        "-c:v", "libx264", # Video codec (widely compatible)
        "-preset", "fast", # Encoding speed preset (good balance)
        "-crf", str(crf_value), # Constant Rate Factor (0=lossless, ~18=visually lossless, 23=default, 51=worst)
        "-c:a", "aac",     # Audio codec (widely compatible)
        "-b:a", "192k",    # Audio bitrate
        "-shortest", # Finish encoding when the shortest input stream ends
        save_path, # Final output path
    ]
    print(f"{log_prefix} Running ffmpeg command: {' '.join(final_command)}") # Log added
    try:
        # Use subprocess.run for better control and error capture
        process = subprocess.run(
            final_command,
            check=True,        # Raise CalledProcessError on failure (non-zero exit code)
            capture_output=True, # Capture stdout and stderr
            text=True          # Decode stdout/stderr as text
        )
        print(f"{log_prefix} FFmpeg command executed successfully.") # Log added
        # Optionally log ffmpeg output for debugging
        # print(f"FFmpeg stdout:\n{process.stdout}")
        # print(f"FFmpeg stderr:\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        # --- Enhanced Error Handling ---
        print(f"{log_prefix} !!! Error during ffmpeg processing (Exit code: {e.returncode})")
        print(f"FFmpeg command failed: {' '.join(e.cmd)}")
        print(f"FFmpeg stdout:\n{e.stdout}")
        print(f"FFmpeg stderr:\n{e.stderr}")
        # Keep the temp file if ffmpeg fails for debugging
        print(f"{log_prefix} FFmpeg failed, temporary file kept at: {save_path_tmp}")
        # Raise a more informative exception
        raise Exception(f"FFmpeg failed to merge video and audio. Check ffmpeg logs above. Temp file: {save_path_tmp}. Error: {e.stderr[:500]}...") # Show first 500 chars of stderr
        # -----------------------------
    except FileNotFoundError:
        # --- Enhanced Error Handling ---
        print(f"{log_prefix} !!! Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")
        raise Exception("ffmpeg not found. Please install ffmpeg and ensure it's in the PATH.")
        # -----------------------------
    except Exception as e:
        # --- Enhanced Error Handling ---
        print(f"{log_prefix} !!! An unexpected error occurred during ffmpeg execution: {e}")
        traceback.print_exc()
        raise # Re-raise
        # -----------------------------


    # --- Clean up temporary file only if ffmpeg succeeded and final file exists ---
    if os.path.exists(save_path):
        try:
            print(f"{log_prefix} Removing temporary file: {save_path_tmp}") # Log added
            os.remove(save_path_tmp)
            print(f"{log_prefix} Removed temporary file successfully.") # Log added
        except OSError as e:
            # This is not critical, just log a warning
            print(f"{log_prefix} Warning: Could not remove temporary file {save_path_tmp}: {e}") # Log added
    else:
         print(f"{log_prefix} Warning: Final output file {save_path} not found after ffmpeg process. Temporary file {save_path_tmp} kept.") # Log added


    # --- Metadata Saving Logic ---
    if save_metadata: # save_metadata comes from args dict
        print(f"{log_prefix} Saving metadata to {metadata_path}...") # Log added
        try:
            end_time = datetime.now()
            end_time_perf = time.perf_counter()
            generation_duration_perf = end_time_perf - start_time_perf
            generation_duration_wall = end_time - start_time

            # Create metadata dictionary - include all args passed initially
            metadata = {
                "generation_timestamp_utc": start_time.utcnow().isoformat() + "Z",
                "generation_duration_seconds": round(generation_duration_perf, 3),
                "generation_wall_time_seconds": round(generation_duration_wall.total_seconds(), 3),
                "output_video_file": os.path.basename(save_path),
                "output_metadata_file": os.path.basename(metadata_path),
                "input_image_file": os.path.basename(args['image_path']),
                "input_audio_file": os.path.basename(args['audio_path']),
                # Include all arguments passed to main for reproducibility
                "settings": args.copy() # Create a copy to avoid modifying original
            }

            # Clean up settings dict: remove non-serializable or redundant items
            # Make sure keys exist before popping
            metadata['settings'].pop('output_dir', None) # Already have output path info
            metadata['settings'].pop('output_base_name', None) # Already have output path info
            metadata['settings'].pop('save_metadata', None) # Redundant within metadata itself
            metadata['settings'].pop('generation_index', None) # Status info, not a setting
            metadata['settings'].pop('total_generations', None) # Status info, not a setting

            # Convert Path objects to strings if any exist in settings
            for key, value in metadata['settings'].items():
                 if isinstance(value, Path):
                      metadata['settings'][key] = str(value)
                 # Optionally remove tensors or large objects if they somehow got passed
                 if isinstance(value, (torch.Tensor, np.ndarray, Image.Image)):
                      metadata['settings'][key] = f"<{type(value).__name__} object>"

            # Write JSON metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            print(f"{log_prefix} Metadata saved successfully.") # Log added

        except Exception as meta_e:
            # --- Enhanced Error Handling ---
            # Log error but don't fail the whole generation
            print(f"{log_prefix} !!! Warning: Failed to save metadata to {metadata_path}: {meta_e}")
            traceback.print_exc()
            # -----------------------------
    # --- End Metadata Saving ---


    print(f"{log_prefix} ===== Finished Generation (Duration: {time.perf_counter() - start_time_perf:.2f}s) =====") # Log added
    return save_path # Return the path to the final generated video

# --- END OF FILE infer.py ---