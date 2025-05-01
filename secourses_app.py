# --- START OF FILE secourses_app.py ---
import platform
import argparse
import sys
import os # Added for path manipulation and folder opening
import subprocess # Added for folder opening
from datetime import datetime
from pathlib import Path
import math
import random
import time # Added for potential delays/checks
import traceback # Added for detailed error logging
import json # Added for batch processing prompt reading
import uuid # Added for unique temporary filenames

import gradio as gr
import librosa
import torch

# Make sure infer.py and its dependencies (like CancelledError) are accessible
try:
    from infer import load_models, main
    # Attempt to import CancelledError, define locally if it fails
    try:
        from diffsynth.pipelines.wan_video import CancelledError
    except ImportError:
        print("Warning: Could not import CancelledError from diffsynth.pipelines.wan_video. Defining locally.")
        class CancelledError(Exception):
            """Custom exception for cancellation (local definition)."""
            pass
except ImportError as e:
     print(f"ERROR: Failed to import 'infer' module or its dependencies: {e}")
     print("Please ensure infer.py and necessary modules (like diffsynth) are in the Python path.")
     sys.exit(1)


# --- Global State Variables ---
pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None
current_torch_dtype = None
current_num_persistent_param_in_dit = None
models_loaded = False

# State flags for UI control and cancellation
is_generating = False # Covers both single and batch generation
is_cancelling = False
cancel_requested = False
# -----------------------------

# --- Constants ---
OUTPUT_DIR = Path("./outputs") # Define output directory consistently
TEMP_AUDIO_DIR = Path("./temp_audios") # Define temporary audio directory

DEFAULT_PROMPT = "A person is talking."
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_FPS = 23
DEFAULT_DURATION = 5
MAX_DURATION = 60 # Keep max duration reasonable
DEFAULT_SEED = 1247
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_PROMPT_CFG = 5.0
DEFAULT_AUDIO_CFG = 5.0
DEFAULT_AUDIO_WEIGHT = 1.0
DEFAULT_SIGMA_SHIFT = 5.0
DEFAULT_DENOISING_STRENGTH = 1.0
DEFAULT_SAVE_QUALITY = 10 # Changed default to CRF 18 (visually lossless standard)
DEFAULT_TILE_SIZE_H = 30
DEFAULT_TILE_SIZE_W = 52
DEFAULT_TILE_STRIDE_H = 15
DEFAULT_TILE_STRIDE_W = 26

MODEL_DIRS = {
    "wan_model_dir": "./models",
    "fantasytalking_model_path": "./models/fantasytalking_model.ckpt",
    "wav2vec_model_dir": "./models/wav2vec2-base-960h",
}

# Updated VRAM Presets (Values are strings to be parsed later)
VRAM_PRESETS = {
    # Name: Value string (commas ok, parsed later) or "" for None
    "6GB GPUs": "0",
    "8GB GPUs": "0",
    "10GB GPUs": "0",
    "12GB GPUs": "0",
    "16GB GPUs": "0", 
    "24GB GPUs": "6,000,000,000",
    "32GB GPUs": "14,000,000,000", 
    "48GB GPUs": "22,000,000,000", 
    "80GB GPUs": "32,000,000,000", 
}
VRAM_PRESET_DEFAULT = "24GB GPUs" # Adjusted default based on new names

TORCH_DTYPES_STR = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
TORCH_DTYPE_DEFAULT = "bfloat16"
# ---------------


def calculate_frames(duration_sec, fps):
    """Calculates the frame count needed for the pipeline (4k+1 format)."""
    if not isinstance(duration_sec, (int, float)) or duration_sec <= 0:
        print(f"Warning: Invalid duration_sec ({duration_sec}), defaulting to 1.")
        duration_sec = 1
    if not isinstance(fps, (int, float)) or fps <= 0:
        print(f"Warning: Invalid fps ({fps}), defaulting to 1.")
        fps = 1
    total_frames = math.ceil(duration_sec * fps)
    # Ensure total_frames is at least 1 for the calculation below
    total_frames = max(1, total_frames)
    # Calculate k for 4k+1 format
    # If total_frames = 1, k = ceil(0/4) = 0. num_frames = 4*0+1 = 1
    # If total_frames = 2, k = ceil(1/4) = 1. num_frames = 4*1+1 = 5
    # If total_frames = 5, k = ceil(4/4) = 1. num_frames = 4*1+1 = 5
    k = math.ceil((total_frames - 1) / 4)
    num_frames = 4 * k + 1
    return num_frames

def parse_persistent_params(vram_custom_value_str):
    """Parses the VRAM custom value string (handles commas, empty string -> None)."""
    num_persistent_param_in_dit = None # Default to None (max persistence)
    if vram_custom_value_str:
        try:
            # Remove commas and whitespace, then try converting to int
            cleaned_value = vram_custom_value_str.replace(',', '').strip()
            if cleaned_value: # Ensure not empty after cleaning
                num_persistent_param_in_dit = int(cleaned_value)
                print(f"[VRAM Setting] Using custom persistent params value: {num_persistent_param_in_dit}")
            else:
                 print("[VRAM Setting] Textbox is empty after cleaning, using None (max persistence).")
        except ValueError:
            print(f"[VRAM Setting] Warning: Invalid custom VRAM value '{vram_custom_value_str}'. Using None (max persistence).")
            gr.Warning(f"Invalid VRAM value '{vram_custom_value_str}'. Using max persistence.")
    else:
         print("[VRAM Setting] Textbox is empty, using None (max persistence).")
    return num_persistent_param_in_dit

def get_torch_dtype(dtype_str):
    """Gets the torch dtype object from its string representation."""
    dtype = TORCH_DTYPES_STR.get(dtype_str, TORCH_DTYPES_STR[TORCH_DTYPE_DEFAULT])
    print(f"[DType Setting] Using torch_dtype: {dtype}")
    return dtype

def update_vram_textbox(preset_name):
    """Updates the VRAM custom value textbox based on the selected preset."""
    value = VRAM_PRESETS.get(preset_name, VRAM_PRESETS[VRAM_PRESET_DEFAULT])
    print(f"[VRAM Setting] Preset '{preset_name}' selected, updating textbox to: '{value}'")
    # Need to return a Gradio component update
    return gr.Textbox(value=value)

def open_folder():
    """Opens the output directory in the default file explorer."""
    output_path_str = str(OUTPUT_DIR.resolve())
    print(f"[File System] Attempting to open output folder: {output_path_str}")
    try:
        if not OUTPUT_DIR.exists():
             print(f"[File System] Output directory does not exist: {output_path_str}")
             gr.Warning(f"Output folder not found: {output_path_str}")
             return # Don't try to open if it doesn't exist

        if sys.platform == "win32":
            os.startfile(output_path_str)
            print("[File System] Opened folder using os.startfile (Windows)")
        elif sys.platform == "darwin":
            subprocess.Popen(["open", output_path_str])
            print("[File System] Opened folder using subprocess.Popen 'open' (macOS)")
        else: # Linux and other Unix-like
            subprocess.Popen(["xdg-open", output_path_str])
            print("[File System] Opened folder using subprocess.Popen 'xdg-open' (Linux)")
    except FileNotFoundError:
         # This might happen if xdg-open or open is not found, though the dir exists
         print(f"[File System] Error: Command (e.g., xdg-open) not found to open {output_path_str}")
         gr.Warning(f"Could not find command to open output folder automatically.")
    except Exception as e:
        print(f"[File System] Error opening folder: {e}")
        traceback.print_exc()
        gr.Warning(f"Could not open output folder: {e}")


# --- Main Generation Function (Handles Single Image Mode) ---
def generate_video(
    # Inputs
    image_path,
    audio_path,
    prompt,
    negative_prompt,
    # Basic Settings
    width,
    height,
    duration_seconds,
    fps,
    num_generations, # Added
    # CFG & Weights
    prompt_cfg_scale,
    audio_cfg_scale,
    audio_weight,
    # Steps & Seed
    inference_steps,
    seed,
    use_random_seed,
    # Advanced
    sigma_shift,
    denoising_strength,
    save_video_quality,
    save_metadata, # Added
    # Performance
    tiled_vae,
    tile_size_h,
    tile_size_w,
    tile_stride_h,
    tile_stride_w,
    vram_custom_value_input_param, # Renamed for clarity (comes from Textbox)
    torch_dtype_str,
    progress=gr.Progress()
):
    """Handles the generation process for single image or multiple sequential generations."""
    global pipe, fantasytalking, wav2vec_processor, wav2vec, models_loaded
    global current_torch_dtype, current_num_persistent_param_in_dit
    global is_generating, is_cancelling, cancel_requested

    # *** ADDED DEBUG PRINT ***
    print(f"\n>>> generate_video called. Current state: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")

    # --- State Check ---
    if is_generating:
        print("[State Check] Generation is already in progress. Ignoring new request.")
        gr.Info("A generation task is already running. Please wait.")
        return None # Return None as no new video is generated
    if is_cancelling:
        print("[State Check] Cancellation is in progress. Ignoring new request.")
        gr.Info("Cancellation is in progress. Please wait before starting a new task.")
        return None
    # -----------------

    # --- Set State ---
    is_generating = True
    # is_cancelling = False # Should be false unless cancel button sets it
    cancel_requested = False # Reset cancel request flag
    print(f"[State Check] Set flags at start: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
    # -----------------

    output_video_path = None # Track the last generated video path

    try:
        # --- Input Validation ---
        progress(0, desc="Validating inputs...")
        print("[Validation] Validating inputs...")
        if image_path is None:
            raise gr.Error("Input Image is required. Please upload an image.")
        if audio_path is None:
            raise gr.Error("Input Audio is required. Please upload or record audio.")

        # Validate dimensions
        try:
             width = int(width)
             height = int(height)
             if width <= 0 or width % 16 != 0:
                  gr.Warning(f"Invalid width ({width}) or not divisible by 16. Using default: {DEFAULT_WIDTH}")
                  width = DEFAULT_WIDTH
             if height <= 0 or height % 16 != 0:
                  gr.Warning(f"Invalid height ({height}) or not divisible by 16. Using default: {DEFAULT_HEIGHT}")
                  height = DEFAULT_HEIGHT
        except (ValueError, TypeError):
             gr.Warning(f"Invalid width/height input. Using defaults: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
             width = DEFAULT_WIDTH
             height = DEFAULT_HEIGHT

        # Validate duration and FPS
        try:
            duration_seconds = float(duration_seconds)
            if duration_seconds <= 0:
                gr.Warning(f"Invalid duration ({duration_seconds}), using default: {DEFAULT_DURATION}s")
                duration_seconds = DEFAULT_DURATION
            duration_seconds = min(duration_seconds, MAX_DURATION) # Apply max duration cap
        except (ValueError, TypeError):
            gr.Warning(f"Invalid duration input. Using default: {DEFAULT_DURATION}s")
            duration_seconds = DEFAULT_DURATION

        try:
            fps = int(fps)
            if fps <= 0:
                 gr.Warning(f"Invalid FPS ({fps}), using default: {DEFAULT_FPS}")
                 fps = DEFAULT_FPS
        except (ValueError, TypeError):
             gr.Warning(f"Invalid FPS input. Using default: {DEFAULT_FPS}")
             fps = DEFAULT_FPS

        # Validate num_generations
        try:
             num_generations = int(num_generations)
             if num_generations < 1:
                  gr.Warning("Number of generations must be at least 1. Setting to 1.")
                  num_generations = 1
        except (ValueError, TypeError):
             gr.Warning("Invalid Number of Generations input. Setting to 1.")
             num_generations = 1

        # Validate seed
        initial_seed = DEFAULT_SEED
        try:
            initial_seed = int(seed)
            if initial_seed < 0:
                 gr.Warning(f"Seed cannot be negative ({initial_seed}). Using default: {DEFAULT_SEED}")
                 initial_seed = DEFAULT_SEED
        except (ValueError, TypeError):
            if not use_random_seed: # Only warn if fixed seed was intended but invalid
                gr.Warning(f"Invalid seed value ('{seed}'). Using default: {DEFAULT_SEED}")
            initial_seed = DEFAULT_SEED # Use default if parsing fails

        print(f"[Validation] Inputs validated. Num Generations={num_generations}, Use Random Seed={use_random_seed}, Initial Seed={initial_seed}")

        # --- Audio Handling (Extraction if Video) ---
        # progress(0.15, desc="Processing audio input...")
        # original_audio_path = audio_path
        # audio_path_to_use = None
        # temp_audio_file = None
        # delete_temp_audio = False
        #
        # # Check if input is a video file based on common video extensions
        # video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
        # input_audio_path_obj = Path(original_audio_path)
        # is_video = input_audio_path_obj.suffix.lower() in video_extensions
        #
        # if is_video:
        #     print(f\"[Audio Proc] Input '{input_audio_path_obj.name}' detected as video. Extracting audio...\")
        #     # Generate a unique temporary filename for the extracted audio
        #     temp_audio_filename = f\"extracted_audio_{uuid.uuid4()}.wav\"
        #     temp_audio_file = TEMP_AUDIO_DIR / temp_audio_filename
        #
        #     # Construct the ffmpeg command
        #     # ... (ffmpeg command removed)
        #     print(f\"[Audio Proc] Running ffmpeg command: {' '.join(ffmpeg_command)}\")
        #     try:
        #         # ... (subprocess.run removed)
        #         if not temp_audio_file.exists() or temp_audio_file.stat().st_size == 0:
        #              raise RuntimeError(\"FFmpeg finished but output file is missing or empty.\")
        #         print(f\"[Audio Proc] Audio successfully extracted to: {temp_audio_file}\")
        #         audio_path_to_use = str(temp_audio_file.resolve())
        #         delete_temp_audio = True # Mark for deletion later
        #     except subprocess.CalledProcessError as e:
        #         # ... (error handling removed)
        #         raise gr.Error(f\"Failed to extract audio from video: {input_audio_path_obj.name}. Check ffmpeg installation and video file. Error: {e.stderr[:500]}...\")
        #     except Exception as e:
        #          # ... (error handling removed)
        #          raise gr.Error(f\"Failed to process video input {input_audio_path_obj.name}. Error: {e}\")
        # else:
        #     print(f\"[Audio Proc] Input '{input_audio_path_obj.name}' detected as audio. Using directly.\")
        #     audio_path_to_use = original_audio_path # Use the original audio path
        audio_path_to_use = audio_path # Assume audio_path is always the correct audio file

        # --- Calculate Target Duration & Frames (using audio_path_to_use) ---
        progress(0.1, desc="Calculating duration...") # Start progress earlier
        target_duration = duration_seconds
        try:
            # Use the potentially extracted audio path here
            actual_audio_duration = librosa.get_duration(filename=audio_path_to_use)
            print(f"[Audio Info] Effective audio duration (from '{Path(audio_path_to_use).name}'): {actual_audio_duration:.2f}s")
            if actual_audio_duration < duration_seconds:
                gr.Warning(f"Requested duration ({duration_seconds}s) is longer than effective audio ({actual_audio_duration:.2f}s). Using audio duration.")
                target_duration = actual_audio_duration
            elif actual_audio_duration <= 0:
                 raise ValueError("Effective audio duration is zero or negative.")

        except Exception as e:
            print(f"[Error] Could not read effective audio file duration: {audio_path_to_use}. Error: {e}")
            traceback.print_exc()
            # Refer to the original input name in the error message
            input_audio_path_obj = Path(audio_path_to_use) # Get path object for name
            raise gr.Error(f"Could not read audio data from input '{input_audio_path_obj.name}'. Please check the file. Error: {e}")

        num_frames = calculate_frames(target_duration, fps)
        print(f"[Calculation] Target duration: {target_duration:.2f}s, Target FPS: {fps}, Calculated num_frames: {num_frames}")
        if num_frames <= 1:
             # Refer to the original input name in the error message
             input_audio_path_obj = Path(audio_path_to_use) # Get path object for name
             raise gr.Error(f"Calculated number of frames for '{input_audio_path_obj.name}' is {num_frames}. Too low, check audio length/FPS.")


        # --- Prepare Paths and Folders ---
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[File System] Ensured output directory exists: {OUTPUT_DIR}")
        try:
            image_path_abs = Path(image_path).resolve().as_posix()
            # Use the potentially extracted audio path here
            audio_path_abs = Path(audio_path_to_use).resolve().as_posix()
        except Exception as e:
             raise gr.Error(f"Invalid input file path provided. Error: {e}")


        # --- Model Loading / Reloading ---
        # Use progress value 0.2 as audio processing is done
        progress(0.2, desc="Checking model status...")
        num_persistent_param_in_dit = parse_persistent_params(vram_custom_value_input_param)
        torch_dtype = get_torch_dtype(torch_dtype_str)

        load_needed = False
        if not models_loaded:
            load_needed = True
            print("[Model Check] Models not loaded yet.")
        elif current_torch_dtype != torch_dtype:
            load_needed = True
            print(f"[Model Check] Torch dtype changed ({current_torch_dtype} -> {torch_dtype}). Reloading models.")
        elif current_num_persistent_param_in_dit != num_persistent_param_in_dit:
            load_needed = True
            # Format None nicely for printing
            current_persist_str = str(current_num_persistent_param_in_dit) if current_num_persistent_param_in_dit is not None else "None"
            new_persist_str = str(num_persistent_param_in_dit) if num_persistent_param_in_dit is not None else "None"
            print(f"[Model Check] VRAM persistence changed ({current_persist_str} -> {new_persist_str}). Reloading models.")

        if load_needed:
            print("[Model Loading] Unloading previous models (if any) and clearing cache...")
            # Explicitly release references
            pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None
            models_loaded = False
            torch.cuda.empty_cache()
            print("[Model Loading] Cache cleared. Starting model load...")
            progress(0.15, desc="Loading models (can take time)...")
            try:
                pipe, fantasytalking, wav2vec_processor, wav2vec = load_models(
                    wan_model_dir=MODEL_DIRS["wan_model_dir"],
                    fantasytalking_model_path=MODEL_DIRS["fantasytalking_model_path"],
                    wav2vec_model_dir=MODEL_DIRS["wav2vec_model_dir"],
                    # Pass the parsed value (int or None)
                    num_persistent_param_in_dit=num_persistent_param_in_dit,
                    torch_dtype=torch_dtype,
                    device="cuda" # Assuming CUDA is desired
                )
                models_loaded = True
                current_torch_dtype = torch_dtype
                current_num_persistent_param_in_dit = num_persistent_param_in_dit
                print("[Model Loading] Models loaded successfully.")
            except Exception as e:
                print(f"[Error] CRITICAL ERROR loading models: {str(e)}")
                traceback.print_exc()
                models_loaded = False
                # Clear potentially partially loaded models
                pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None
                torch.cuda.empty_cache()
                raise gr.Error(f"Failed to load models. Check paths, VRAM, and console logs. Error: {str(e)}")
        else:
            print("[Model Check] Models already loaded and configuration matches. Skipping reload.")


        # --- Generation Loop ---
        print(f"[Generation] Starting generation loop for {num_generations} video(s)...")
        # Define cancel function based on global flag
        cancel_fn = lambda: cancel_requested

        for i in range(num_generations):
            current_gen_index = i + 1
            progress_desc = f"Generating video {current_gen_index}/{num_generations}..."
            progress((i / num_generations) * 0.6 + 0.3, desc=progress_desc) # Scale progress: 0.3 to 0.9

            # --- Cancellation Check (before starting expensive work) ---
            if cancel_requested:
                print(f"[Cancellation] Cancellation detected before starting generation {current_gen_index}.")
                gr.Warning("Generation cancelled by user.")
                break # Exit the loop

            # --- Determine Seed for Current Generation ---
            current_seed = 0
            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
                print(f"[Seed Info][Gen {current_gen_index}/{num_generations}] Using random seed: {current_seed}")
            else:
                current_seed = initial_seed + i
                print(f"[Seed Info][Gen {current_gen_index}/{num_generations}] Using fixed seed: {current_seed} (Initial: {initial_seed} + {i})")

            # --- Prepare Arguments for infer.main ---
            print(f"[Args Prep][Gen {current_gen_index}/{num_generations}] Preparing arguments for infer.main...")
            args_dict = {
                "image_path": image_path_abs,
                "audio_path": audio_path_abs,
                "prompt": prompt if prompt else DEFAULT_PROMPT,
                "negative_prompt": negative_prompt,
                "output_dir": str(OUTPUT_DIR),
                "width": width,
                "height": height,
                "num_frames": num_frames, # Use calculated num_frames
                "fps": fps,
                "audio_weight": float(audio_weight),
                "prompt_cfg_scale": float(prompt_cfg_scale),
                "audio_cfg_scale": float(audio_cfg_scale),
                "inference_steps": int(inference_steps),
                "seed": current_seed, # Use calculated seed
                "tiled_vae": bool(tiled_vae),
                "tile_size_h": int(tile_size_h),
                "tile_size_w": int(tile_size_w),
                "tile_stride_h": int(tile_stride_h),
                "tile_stride_w": int(tile_stride_w),
                "sigma_shift": float(sigma_shift),
                "denoising_strength": float(denoising_strength),
                "save_video_quality": int(save_video_quality),
                "save_metadata": bool(save_metadata), # Pass metadata flag
                # Control info for logging/naming in infer.py
                "generation_index": current_gen_index,
                "total_generations": num_generations,
                "output_base_name": None, # Use None for sequential naming (0001, 0002...)
            }

            # --- Execute Generation ---
            print(f"[Execution][Gen {current_gen_index}/{num_generations}] Calling infer.main...")
            try:
                # *** Pass cancel_fn to main ***
                output_video_path = main(
                    args_dict, pipe, fantasytalking, wav2vec_processor, wav2vec, cancel_fn=cancel_fn
                )
                print(f"[Execution][Gen {current_gen_index}/{num_generations}] infer.main completed. Output: {output_video_path}")
                # Update progress after successful generation
                progress(((i + 1) / num_generations) * 0.6 + 0.3, desc=f"Finished video {current_gen_index}/{num_generations}")

            except CancelledError as ce:
                print(f"[Cancellation][Gen {current_gen_index}/{num_generations}] Caught CancelledError from infer.main: {ce}")
                gr.Warning(f"Generation {current_gen_index}/{num_generations} cancelled by user.")
                # Attempt to unload models cleanly after cancellation
                if pipe is not None and hasattr(pipe, 'load_models_to_device') and callable(getattr(pipe, 'load_models_to_device')):
                    print("[Cancellation] Attempting to unload models from GPU due to cancellation...")
                    try:
                        # Assuming [] triggers offload based on wan_video.py logic
                        pipe.load_models_to_device([])
                        print("[Cancellation] Models unloaded/offloaded.")
                    except Exception as unload_e:
                        print(f"[Warning] Error during model unloading on cancel: {unload_e}")
                # *** Break the loop after handling cancellation ***
                break # Stop generation loop immediately on cancel

            except Exception as e:
                print(f"[Error][Gen {current_gen_index}/{num_generations}] An error occurred during infer.main: {str(e)}")
                print(f"Exception type: {type(e)}")
                traceback.print_exc()
                # Attempt to clear cache
                print("[Error] Attempting to clear CUDA cache after error...")
                torch.cuda.empty_cache()
                print("[Error] CUDA cache clear attempted.")
                # Raise a Gradio error to notify the user, stopping the loop
                raise gr.Error(f"Error during video generation {current_gen_index}/{num_generations}: {str(e)}")

        # --- Loop Finished ---
        if not cancel_requested:
             progress(1.0, desc="All generations complete!")
             print("[Generation] Generation loop finished.")
        else:
             print("[Generation] Generation loop exited due to cancellation.")


        # Return the path of the *last* successfully generated video
        return output_video_path


    except gr.Error as gr_e: # Catch Gradio-specific errors (validation, etc.)
         print(f"[Gradio Error] {gr_e}")
         # No need to raise again, Gradio handles it
         return None # Indicate failure
    except Exception as e: # Catch unexpected errors
         print(f"[Error] An unexpected error occurred in generate_video: {str(e)}")
         traceback.print_exc()
         gr.Error(f"An unexpected error occurred: {str(e)}") # Show error in UI
         return None # Indicate failure
    finally:
        # --- Reset State ---
        # *** ADDED DEBUG PRINTS ***
        print(f"[State Check] Entering FINALLY block for generate_video. Current state: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
        is_generating = False
        is_cancelling = False # Make absolutely sure this is reset
        cancel_requested = False # Reset request flag too
        print(f"[State Check] Exiting FINALLY block for generate_video. Flags after reset: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
        # Optional Cache Clear
        # print("Clearing CUDA cache in finally block...")
        # torch.cuda.empty_cache()
        # print("CUDA cache cleared.")

# --- Function to Handle Video Upload and Audio Extraction ---
def handle_video_upload(video_file_path, progress=gr.Progress()):
    """Extracts audio from video, saves to temp, returns path for gr.Audio update."""
    if video_file_path is None:
        print("[Video Handler] No video file provided.")
        # Return None or current value? Return None might clear audio input if user cancels.
        # Let's return an update to clear if no file is provided (or selection is cleared)
        return gr.Audio(value=None)

    progress(0, desc="Extracting audio from video...")
    print(f"[Video Handler] Received video file: {video_file_path}")
    video_path_obj = Path(video_file_path)
    extracted_audio_path = None

    # Generate a unique temporary filename for the extracted audio
    temp_audio_filename = f"extracted_audio_{uuid.uuid4()}.wav"
    temp_audio_file = TEMP_AUDIO_DIR / temp_audio_filename

    # Construct the ffmpeg command (similar to before, force mono WAV)
    ffmpeg_command = [
        'ffmpeg',
        '-i', str(video_path_obj.resolve()),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '1', # Mono
        '-y',
        str(temp_audio_file.resolve())
    ]

    print(f"[Video Handler] Running ffmpeg command: {' '.join(ffmpeg_command)}")
    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, timeout=60) # Added timeout
        print(f"[Video Handler] FFmpeg stdout: {process.stdout}")
        print(f"[Video Handler] FFmpeg stderr: {process.stderr}")
        if not temp_audio_file.exists() or temp_audio_file.stat().st_size == 0:
             raise RuntimeError("FFmpeg finished but output file is missing or empty.")

        extracted_audio_path = str(temp_audio_file.resolve())
        print(f"[Video Handler] Audio successfully extracted to: {extracted_audio_path}")
        progress(1, desc="Audio extracted!")
        gr.Info(f"Audio extracted from {video_path_obj.name} and loaded.")
        # Return the path to update the gr.Audio component
        return gr.Audio(value=extracted_audio_path)

    except subprocess.TimeoutExpired:
        print(f"[Error][Video Handler] FFmpeg command timed out after 60 seconds.")
        gr.Warning("Audio extraction took too long and was cancelled.")
        # Clean up potentially partially created file
        if temp_audio_file.exists(): temp_audio_file.unlink()
        return gr.Audio(value=None) # Clear audio input on error
    except subprocess.CalledProcessError as e:
        print(f"[Error][Video Handler] FFmpeg failed with exit code {e.returncode}")
        print(f"[Error][Video Handler] FFmpeg stderr: {e.stderr}")
        gr.Error(f"Failed to extract audio from {video_path_obj.name}. Error: {e.stderr[:500]}...")
        return gr.Audio(value=None) # Clear audio input on error
    except Exception as e:
         print(f"[Error][Video Handler] An unexpected error occurred during audio extraction: {e}")
         traceback.print_exc()
         gr.Error(f"Failed to process video {video_path_obj.name}. Error: {e}")
         return gr.Audio(value=None) # Clear audio input on error


# --- Batch Processing Function ---
def process_batch(
    # Batch Specific Inputs
    batch_input_folder_str,
    batch_output_folder_str,
    batch_skip_existing,
    batch_use_gradio_audio,
    batch_use_gradio_prompt,
    # Inputs from UI (used as fallbacks or base settings)
    image_input_fallback, # Not directly used, but good practice
    audio_input_fallback,
    prompt_fallback,
    negative_prompt,
    # Basic Settings
    width,
    height,
    duration_seconds,
    fps,
    # CFG & Weights
    prompt_cfg_scale,
    audio_cfg_scale,
    audio_weight,
    # Steps & Seed
    inference_steps,
    seed,
    use_random_seed,
    # Advanced
    sigma_shift,
    denoising_strength,
    save_video_quality,
    save_metadata,
    # Performance
    tiled_vae,
    tile_size_h,
    tile_size_w,
    tile_stride_h,
    tile_stride_w,
    vram_custom_value_input_param,
    torch_dtype_str,
    progress=gr.Progress()
):
    """Handles the batch processing of images from a folder."""
    global pipe, fantasytalking, wav2vec_processor, wav2vec, models_loaded
    global current_torch_dtype, current_num_persistent_param_in_dit
    global is_generating, is_cancelling, cancel_requested

    # *** ADDED DEBUG PRINT ***
    print(f"\n>>> process_batch called. Current state: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")

    # --- State Check ---
    if is_generating:
        print("[State Check][Batch] Generation/Batch is already in progress. Ignoring batch request.")
        gr.Info("A generation or batch task is already running. Please wait.")
        return None
    if is_cancelling:
        print("[State Check][Batch] Cancellation is in progress. Ignoring batch request.")
        gr.Info("Cancellation is in progress. Please wait before starting a new task.")
        return None
    # -----------------

    # --- Set State ---
    is_generating = True # Use the same flag as single generation
    # is_cancelling = False # Should be false unless cancel button sets it
    cancel_requested = False
    print(f"[State Check][Batch] Set flags at start: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
    # -----------------

    processed_files_count = 0
    skipped_files_count = 0
    error_files_count = 0
    last_output_path = None # Track last successful output for potential return

    try:
        # --- Validate Batch Inputs ---
        progress(0, desc="Validating batch inputs...")
        print("[Validation][Batch] Validating batch inputs...")
        if not batch_input_folder_str or not batch_output_folder_str:
            raise gr.Error("Batch Input and Output folders must be specified.")

        batch_input_folder = Path(batch_input_folder_str)
        batch_output_folder = Path(batch_output_folder_str)

        if not batch_input_folder.is_dir():
            raise gr.Error(f"Batch Input folder not found or is not a directory: {batch_input_folder}")

        # Ensure output folder exists
        batch_output_folder.mkdir(parents=True, exist_ok=True)
        print(f"[File System][Batch] Ensured batch output directory exists: {batch_output_folder}")

        # --- Find Image Files ---
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        print(f"[File System][Batch] Scanning for images ({', '.join(image_extensions)}) in: {batch_input_folder}")
        image_files = sorted([p for p in batch_input_folder.glob("*") if p.suffix.lower() in image_extensions])

        if not image_files:
            raise gr.Error(f"No images found in the batch input folder: {batch_input_folder}")

        total_files = len(image_files)
        print(f"[File System][Batch] Found {total_files} image file(s).")

        # --- Validate other inputs (same as single generation) ---
        # (Re-using validation logic from generate_video - consider refactoring later if needed)
        try:
             width = int(width); height = int(height)
             if width <= 0 or width % 16 != 0: width = DEFAULT_WIDTH; gr.Warning(f"[Batch] Invalid width, using default {DEFAULT_WIDTH}")
             if height <= 0 or height % 16 != 0: height = DEFAULT_HEIGHT; gr.Warning(f"[Batch] Invalid height, using default {DEFAULT_HEIGHT}")
        except: width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT; gr.Warning(f"[Batch] Invalid dims, using defaults")
        try:
             duration_seconds = float(duration_seconds); duration_seconds = max(1, min(duration_seconds, MAX_DURATION))
        except: duration_seconds = DEFAULT_DURATION; gr.Warning(f"[Batch] Invalid duration, using default")
        try: fps = int(fps); fps = max(1, fps)
        except: fps = DEFAULT_FPS; gr.Warning(f"[Batch] Invalid FPS, using default")
        initial_seed = DEFAULT_SEED
        try: initial_seed = int(seed); initial_seed = max(0, initial_seed)
        except: initial_seed = DEFAULT_SEED; gr.Warning(f"[Batch] Invalid seed, using default")


        # --- Model Loading / Reloading Check (same as single generation) ---
        progress(0.1, desc="Checking model status...")
        num_persistent_param_in_dit = parse_persistent_params(vram_custom_value_input_param)
        torch_dtype = get_torch_dtype(torch_dtype_str)

        load_needed = False
        if not models_loaded: load_needed = True; print("[Model Check][Batch] Models not loaded yet.")
        elif current_torch_dtype != torch_dtype: load_needed = True; print(f"[Model Check][Batch] DType changed. Reloading.")
        elif current_num_persistent_param_in_dit != num_persistent_param_in_dit: load_needed = True; print(f"[Model Check][Batch] VRAM persistence changed. Reloading.")

        if load_needed:
            print("[Model Loading][Batch] Unloading/Reloading models...")
            pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None; models_loaded = False
            torch.cuda.empty_cache()
            progress(0.15, desc="Loading models for batch...")
            try:
                pipe, fantasytalking, wav2vec_processor, wav2vec = load_models(MODEL_DIRS["wan_model_dir"], MODEL_DIRS["fantasytalking_model_path"], MODEL_DIRS["wav2vec_model_dir"], num_persistent_param_in_dit, torch_dtype, "cuda")
                models_loaded = True; current_torch_dtype = torch_dtype; current_num_persistent_param_in_dit = num_persistent_param_in_dit
                print("[Model Loading][Batch] Models loaded.")
            except Exception as e: print(f"[Error][Batch] CRITICAL ERROR loading models: {e}"); traceback.print_exc(); raise gr.Error(f"Failed to load models for batch. Error: {e}")
        else:
            print("[Model Check][Batch] Models ready.")


        # --- Batch Processing Loop ---
        print(f"[Batch] Starting batch processing loop for {total_files} image(s)...")
        cancel_fn = lambda: cancel_requested # Define cancel function

        for i, image_path in enumerate(image_files):
            current_batch_index = i + 1
            progress_desc = f"Batch {current_batch_index}/{total_files}: {image_path.name}"
            progress((i / total_files) * 0.8 + 0.15, desc=progress_desc) # Scale progress: 0.15 to 0.95

            # --- Cancellation Check ---
            if cancel_requested:
                print(f"[Cancellation][Batch] Cancellation detected before processing item {current_batch_index}.")
                gr.Warning("Batch processing cancelled by user.")
                break

            print(f"\n--- Processing Batch Item {current_batch_index}/{total_files}: {image_path.name} ---")
            image_stem = image_path.stem
            output_video_path = batch_output_folder / f"{image_stem}.mp4"
            output_metadata_path = batch_output_folder / f"{image_stem}.txt"

            # --- Skip Logic ---
            if batch_skip_existing:
                video_exists = output_video_path.exists()
                metadata_exists = save_metadata and output_metadata_path.exists()
                if video_exists or metadata_exists:
                    reason = []
                    if video_exists: reason.append("video exists")
                    if metadata_exists: reason.append("metadata exists")
                    print(f"[Skip][Batch] Skipping '{image_path.name}' because {' and '.join(reason)}.")
                    skipped_files_count += 1
                    continue # Skip to the next file

            # --- Find Corresponding Audio ---
            audio_file_to_use = None
            audio_extensions = {".wav", ".mp3", ".flac"} # Add more if needed
            found_audio = False
            for ext in audio_extensions:
                 potential_audio_path = batch_input_folder / f"{image_stem}{ext}"
                 if potential_audio_path.exists():
                      audio_file_to_use = potential_audio_path.resolve().as_posix()
                      print(f"[File Match][Batch] Found matching audio: {potential_audio_path.name}")
                      found_audio = True
                      break
            if not found_audio:
                 if batch_use_gradio_audio and audio_input_fallback:
                      audio_file_to_use = Path(audio_input_fallback).resolve().as_posix()
                      print(f"[File Match][Batch] No matching audio found for '{image_stem}'. Using UI audio: {Path(audio_file_to_use).name}")
                 else:
                      print(f"[Error][Batch] No matching audio found for '{image_stem}' and UI fallback disabled or missing. Skipping.")
                      gr.Warning(f"Skipping {image_path.name}: No matching audio found.")
                      error_files_count += 1
                      continue # Skip this image

            # --- Find Corresponding Prompt ---
            prompt_to_use = prompt_fallback if prompt_fallback else DEFAULT_PROMPT # Default
            prompt_source = "UI fallback or default"
            potential_prompt_path = batch_input_folder / f"{image_stem}.txt"
            if potential_prompt_path.exists():
                try:
                    with open(potential_prompt_path, 'r', encoding='utf-8') as f:
                        prompt_to_use = f.read().strip()
                        prompt_source = f"matched file ({potential_prompt_path.name})"
                    print(f"[File Match][Batch] Found and read matching prompt: {potential_prompt_path.name}")
                except Exception as e:
                    print(f"[Warning][Batch] Found prompt file {potential_prompt_path.name} but failed to read: {e}. Using {prompt_source}.")
                    gr.Warning(f"Error reading prompt file {potential_prompt_path.name}. Using fallback/default.")
            elif not batch_use_gradio_prompt:
                 prompt_source = "default (UI fallback disabled)"
                 prompt_to_use = DEFAULT_PROMPT # Force default if fallback disabled

            print(f"[Prompt Info][Batch] Using prompt from: {prompt_source}")


            # --- Calculate Duration & Frames for this item ---
            current_target_duration = duration_seconds
            try:
                item_audio_duration = librosa.get_duration(filename=audio_file_to_use)
                print(f"[Audio Info][Batch] Item '{image_stem}' audio duration: {item_audio_duration:.2f}s")
                if item_audio_duration < duration_seconds:
                    print(f"[Info][Batch] Item '{image_stem}' audio ({item_audio_duration:.2f}s) is shorter than requested max duration ({duration_seconds}s). Using audio duration.")
                    current_target_duration = item_audio_duration
                elif item_audio_duration <= 0:
                     raise ValueError("Audio duration is zero or negative.")
            except Exception as e:
                 print(f"[Error][Batch] Failed to get duration for audio '{Path(audio_file_to_use).name}'. Skipping item '{image_stem}'. Error: {e}")
                 gr.Warning(f"Skipping {image_path.name}: Error reading audio duration.")
                 error_files_count += 1
                 continue

            current_num_frames = calculate_frames(current_target_duration, fps)
            print(f"[Calculation][Batch] Item '{image_stem}' target duration: {current_target_duration:.2f}s, num_frames: {current_num_frames}")
            if current_num_frames <= 1:
                 print(f"[Error][Batch] Calculated frames for '{image_stem}' is {current_num_frames}. Too low. Skipping.")
                 gr.Warning(f"Skipping {image_path.name}: Calculated frames too low ({current_num_frames}).")
                 error_files_count += 1
                 continue


            # --- Determine Seed ---
            current_seed = 0
            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = initial_seed + i # Increment seed based on position in *found* image list

            print(f"[Seed Info][Batch] Item '{image_stem}' using seed: {current_seed}")

            # --- Prepare Arguments for infer.main ---
            print(f"[Args Prep][Batch] Preparing arguments for item '{image_stem}'...")
            args_dict = {
                "image_path": image_path.resolve().as_posix(),
                "audio_path": audio_file_to_use, # Already resolved path
                "prompt": prompt_to_use,
                "negative_prompt": negative_prompt,
                "output_dir": str(batch_output_folder), # Use batch output dir
                "width": width,
                "height": height,
                "num_frames": current_num_frames, # Use item-specific frame count
                "fps": fps,
                "audio_weight": float(audio_weight),
                "prompt_cfg_scale": float(prompt_cfg_scale),
                "audio_cfg_scale": float(audio_cfg_scale),
                "inference_steps": int(inference_steps),
                "seed": current_seed,
                "tiled_vae": bool(tiled_vae),
                "tile_size_h": int(tile_size_h), "tile_size_w": int(tile_size_w),
                "tile_stride_h": int(tile_stride_h), "tile_stride_w": int(tile_stride_w),
                "sigma_shift": float(sigma_shift),
                "denoising_strength": float(denoising_strength),
                "save_video_quality": int(save_video_quality),
                "save_metadata": bool(save_metadata),
                # Control info for logging/naming
                "generation_index": current_batch_index,
                "total_generations": total_files,
                "output_base_name": image_stem, # *** Use image stem for naming ***
            }

            # --- Execute Generation for Batch Item ---
            try:
                # Add logging for the output directory being passed
                print(f"[Execution][Batch] Calling infer.main for '{image_stem}' with output_dir: '{args_dict['output_dir']}'")
                # Pass cancel_fn to main
                last_output_path = main(args_dict, pipe, fantasytalking, wav2vec_processor, wav2vec, cancel_fn=cancel_fn)
                print(f"[Execution][Batch] Successfully processed item '{image_stem}'. Output: {last_output_path}")
                processed_files_count += 1
            except CancelledError as ce:
                print(f"[Cancellation][Batch] Caught CancelledError during item '{image_stem}': {ce}")
                gr.Warning(f"Batch processing cancelled by user during item {current_batch_index}.")
                # Attempt model unload in finally block
                break # Stop the batch loop
            except Exception as e:
                print(f"[Error][Batch] Failed to process item '{image_stem}'. Error: {str(e)}")
                traceback.print_exc()
                gr.Warning(f"Error processing {image_path.name}: {e}. Skipping.")
                error_files_count += 1
                # Clear cache potentially? Optional.
                # torch.cuda.empty_cache()
                continue # Continue to the next item in the batch

        # --- Batch Loop Finished ---
        if not cancel_requested:
             final_desc = f"Batch complete! Processed: {processed_files_count}, Skipped: {skipped_files_count}, Errors: {error_files_count}."
             progress(1.0, desc=final_desc)
             print(f"[Batch] Batch processing finished. Processed: {processed_files_count}, Skipped: {skipped_files_count}, Errors: {error_files_count}.")
             gr.Info(final_desc)
        else:
             print("[Batch] Batch processing loop exited due to cancellation.")
             gr.Warning(f"Batch cancelled. Processed: {processed_files_count}, Skipped: {skipped_files_count}, Errors: {error_files_count} before cancel.")


        # Return the path of the last successfully generated video in the batch
        return last_output_path

    except gr.Error as gr_e:
         print(f"[Gradio Error][Batch] {gr_e}")
         return None
    except Exception as e:
         print(f"[Error][Batch] An unexpected error occurred in process_batch: {str(e)}")
         traceback.print_exc()
         gr.Error(f"An unexpected batch error occurred: {str(e)}")
         return None
    finally:
        # --- Reset State ---
        # *** ADDED DEBUG PRINTS ***
        print(f"[State Check] Entering FINALLY block for process_batch. Current state: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
        is_generating = False
        is_cancelling = False # Explicitly reset
        cancel_requested = False # Explicitly reset
        print(f"[State Check] Exiting FINALLY block for process_batch. Flags after reset: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
        # --- Unload models if cancelled? ---
        # Decide if unloading here is appropriate or if it should only happen in generate_video's finally
        # If process_batch is cancelled, maybe models should unload too.
        # if cancel_requested and pipe is not None: # Check if cancel caused exit
        #      print("[Cleanup][Batch] Attempting model unload after potentially cancelled batch...")
        #      try: pipe.load_models_to_device([]); print("[Cleanup][Batch] Models unloaded.")
        #      except Exception as ue: print(f"[Warning] Error unloading model post-batch-cancel: {ue}")


# --- Cancel Handler ---
# *** REFINED VERSION ***
def handle_cancel():
    """Sets the cancellation flag and provides user feedback."""
    global is_generating, is_cancelling, cancel_requested # Make sure all relevant globals are listed

    # Check if we are actually running something OR if already cancelling
    if not is_generating or is_cancelling:
        print("[Cancel Handler] No active generation/batch to cancel, or cancellation already in progress.")
        # Optionally provide feedback, but avoid spamming if clicked repeatedly when idle
        # gr.Info("Nothing to cancel or cancellation already requested.")
        return # Do nothing if not generating or already cancelling

    print("[Cancel Handler] Cancel button clicked. Setting flags.")
    cancel_requested = True # Signal the running function to stop
    is_cancelling = True    # Set state to prevent new generations and repeated cancel signals
    gr.Warning("Cancel requested! Attempting to stop generation...")
    # Gradio's `cancels` mechanism will now try to interrupt the target function.
    # The finally block in the main functions will reset is_cancelling and is_generating later.


# --- Gradio UI Definition ---
with gr.Blocks(title="FantasyTalking Video Generation (SECourses App V1)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # FantasyTalking: Realistic Talking Portrait Generation SECourses App V3 - https://www.patreon.com/posts/127855145
    Generate a talking head video from an image and audio, or process a batch of images.
    [GitHub](https://github.com/Fantasy-AMAP/fantasy-talking) | [arXiv Paper](https://arxiv.org/abs/2504.04842)
    """
    )

    with gr.Row():
        # --- Left Column: Inputs & Basic Settings ---
        with gr.Column(scale=2):
            gr.Markdown("## 1. Inputs (Single Generation)")
            with gr.Row():
                image_input = gr.Image(label="Input Image", type="filepath", scale=1)
                # audio_input = gr.File(label="Input Audio or Video", file_types=["audio", ".wav", ".mp3", ".flac", "video", ".mp4", ".mov", ".avi", ".mkv"], type="filepath", scale=1)
                audio_input = gr.Audio(label="Input Audio (WAV/MP3)", type="filepath", scale=1) # Reverted to gr.Audio

            # Add a separate input for video uploads
            video_audio_input = gr.File(
                label="Or Upload Video to Extract Audio",
                file_types=["video", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"],
                type="filepath",height=150
            )

            prompt_input = gr.Textbox(
                label="Prompt", placeholder=DEFAULT_PROMPT, value=DEFAULT_PROMPT, lines=2,
                info="Describe the scene/action. Used for single gen or batch fallback."
            )
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt", placeholder=DEFAULT_NEGATIVE_PROMPT, value=DEFAULT_NEGATIVE_PROMPT, lines=2,
                info="Describe what NOT to generate. Used for single and batch."
            )

            gr.Examples(
                examples=[
                    ["assets/images/woman.png", "assets/audios/woman.wav", "A woman is talking."]
                ],
                inputs=[image_input, audio_input, prompt_input],
                label="Examples (Image, Audio, Prompt)"
            )

            gr.Markdown("## 2. Generation Settings")
            with gr.Row():
                 width_input = gr.Number(value=DEFAULT_WIDTH, label="Width", minimum=64, step=16, precision=0, info="Must be divisible by 16")
                 height_input = gr.Number(value=DEFAULT_HEIGHT, label="Height", minimum=64, step=16, precision=0, info="Must be divisible by 16")
            with gr.Row():
                duration_input = gr.Number(value=DEFAULT_DURATION, minimum=1, maximum=MAX_DURATION, label="Max Duration (s)", info=f"Video length (capped by audio, max {MAX_DURATION}s)")
                fps_input = gr.Number(value=DEFAULT_FPS, minimum=1, maximum=60, label="FPS", precision=0)

            num_generations_input = gr.Number(label="Number of Generations (Single Mode)", value=1, minimum=1, step=1, precision=0, info="Generate multiple sequential videos with varying seeds.")

            with gr.Row():
                prompt_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, value=DEFAULT_PROMPT_CFG, step=0.5, label="Prompt CFG")
                audio_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, value=DEFAULT_AUDIO_CFG, step=0.5, label="Audio CFG")
            with gr.Row():
                audio_weight = gr.Slider(minimum=0.1, maximum=3.0, value=DEFAULT_AUDIO_WEIGHT, step=0.1, label="Audio Weight", info="Overall audio influence strength.")
                inference_steps = gr.Slider(minimum=1, maximum=100, value=DEFAULT_INFERENCE_STEPS, step=1, label="Inference Steps")

            with gr.Row():
                seed_input = gr.Number(value=DEFAULT_SEED, label="Seed", minimum=-1, precision=0, info="-1 for default")
                random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=True, info="Overrides Seed if checked.")

            with gr.Accordion("Advanced Settings", open=True):
                with gr.Row():
                    sigma_shift = gr.Slider(minimum=0.1, maximum=10.0, value=DEFAULT_SIGMA_SHIFT, step=0.1, label="Sigma Shift (Scheduler)")
                    denoising_strength = gr.Slider(minimum=0.1, maximum=1.0, value=DEFAULT_DENOISING_STRENGTH, step=0.05, label="Denoising Strength", info="Usually 1.0 for I2V from image.")
                with gr.Row():
                    save_video_quality = gr.Slider(minimum=0, maximum=51, value=DEFAULT_SAVE_QUALITY, step=1, label="Output Quality (CRF)", info="Lower CRF = higher quality (0=lossless, ~18=good, 23=default, 51=worst).")
                    save_metadata_checkbox = gr.Checkbox(label="Save Generation Metadata (.txt)", value=True, info="Save settings, timings, filenames.")

            with gr.Accordion("Performance & VRAM", open=True):
                gr.Markdown("_Higher resolution/duration needs more VRAM. Precision affects speed/VRAM._")
                # VRAM Preset Dropdown (Helper)
                with gr.Row():
                    vram_preset_dropdown = gr.Dropdown(
                        choices=list(VRAM_PRESETS.keys()),
                        value=VRAM_PRESET_DEFAULT,
                        label="VRAM Usage Preset Helper",
                        info="Select a preset to auto-fill the value below."
                    )
                    # Actual VRAM Value Input (Textbox)
                    vram_custom_value_input = gr.Textbox(
                        label="Persistent Params Value",
                        value=VRAM_PRESETS[VRAM_PRESET_DEFAULT],
                        info="Effective value (number of parameters kept in VRAM). 0 is slowest bust least VRAM usage. Values are set for 512px and 5 sec vidoes. More seconds = more VRAM demanding, more resolution = more VRAM demanding"
                     )
                # Link dropdown to textbox update
                vram_preset_dropdown.change(fn=update_vram_textbox, inputs=vram_preset_dropdown, outputs=vram_custom_value_input)
                with gr.Row():
                    torch_dtype_dropdown = gr.Dropdown(
                        choices=list(TORCH_DTYPES_STR.keys()), value=TORCH_DTYPE_DEFAULT, label="Computation Precision (dtype)",
                        info="bfloat16 recommended if supported. Requires model reload on change."
                    )
                    tiled_vae_checkbox = gr.Checkbox(label="Enable Tiled VAE", value=True, info="Saves VRAM during decode, might be slightly slower.")
                with gr.Group(visible=True) as tile_options: # Keep tile options always visible but respect checkbox
                    # Tiling options enabled/disabled based on checkbox in backend logic
                    with gr.Row():
                        tile_size_h_input = gr.Number(value=DEFAULT_TILE_SIZE_H, label="Tile H", precision=0)
                        tile_size_w_input = gr.Number(value=DEFAULT_TILE_SIZE_W, label="Tile W", precision=0)
                    with gr.Row():
                        tile_stride_h_input = gr.Number(value=DEFAULT_TILE_STRIDE_H, label="Stride H", precision=0)
                        tile_stride_w_input = gr.Number(value=DEFAULT_TILE_STRIDE_W, label="Stride W", precision=0)


        # --- Right Column: Output & Batch ---
        with gr.Column(scale=1):
            gr.Markdown("## 3. Execution & Output")
            with gr.Row():
                process_btn = gr.Button("Generate Single Video", variant="primary", scale=2)
                cancel_btn = gr.Button("Cancel All", variant="stop", scale=1) # Single cancel button
            video_output = gr.Video(label="Generated Video Output")
            open_folder_btn = gr.Button("Open Output Folder")

            with gr.Accordion("Batch Processing", open=True):
                 gr.Markdown(
                     """
                     Process multiple images from a folder. It looks for matching audio (`<name>.wav/mp3`)
                     and prompts (`<name>.txt`) in the *same folder* as the images.
                     Uses UI settings as defaults or fallbacks if enabled.
                     """
                 )
                 batch_input_folder_input = gr.Textbox(label="Batch Input Folder (contains images, audios, txts)", placeholder="/path/to/your/inputs")
                 batch_output_folder_input = gr.Textbox(label="Batch Output Folder (where videos are saved)", placeholder="/path/to/your/outputs", value=str(OUTPUT_DIR)) # Default to main output dir
                 with gr.Row():
                      batch_skip_existing_checkbox = gr.Checkbox(label="Skip if Output Exists", value=True)
                      batch_use_gradio_audio_checkbox = gr.Checkbox(label="Use UI Audio if match not found", value=True)
                      batch_use_gradio_prompt_checkbox = gr.Checkbox(label="Use UI Prompt if match not found", value=True)

                 batch_start_btn = gr.Button("Start Batch Process", variant="primary")
                 # Cancel button is shared

    # --- Event Handling ---

    # Single Generation Button
    gen_event = process_btn.click(
        fn=generate_video,
        inputs=[
            image_input, audio_input, prompt_input, negative_prompt_input,
            width_input, height_input, duration_input, fps_input, num_generations_input,
            prompt_cfg_scale, audio_cfg_scale, audio_weight,
            inference_steps, seed_input, random_seed_checkbox,
            sigma_shift, denoising_strength, save_video_quality, save_metadata_checkbox,
            tiled_vae_checkbox, tile_size_h_input, tile_size_w_input, tile_stride_h_input, tile_stride_w_input,
            vram_custom_value_input, # Use the textbox value
            torch_dtype_dropdown,
        ],
        outputs=video_output,
    )

    # Batch Generation Button
    batch_event = batch_start_btn.click(
         fn=process_batch,
         inputs=[
              # Batch specific
              batch_input_folder_input, batch_output_folder_input,
              batch_skip_existing_checkbox, batch_use_gradio_audio_checkbox, batch_use_gradio_prompt_checkbox,
              # Fallbacks / General Settings from UI
              image_input, audio_input, prompt_input, negative_prompt_input,
              width_input, height_input, duration_input, fps_input,
              prompt_cfg_scale, audio_cfg_scale, audio_weight,
              inference_steps, seed_input, random_seed_checkbox,
              sigma_shift, denoising_strength, save_video_quality, save_metadata_checkbox,
              tiled_vae_checkbox, tile_size_h_input, tile_size_w_input, tile_stride_h_input, tile_stride_w_input,
              vram_custom_value_input, # Use the textbox value
              torch_dtype_dropdown,
         ],
         outputs=video_output # Show last generated batch video
    )

    # Universal Cancel Button
    # It calls the handle_cancel function *immediately* to set flags
    # And uses `cancels` to signal Gradio to interrupt the running events
    cancel_btn.click(
        fn=handle_cancel,        # Function to set flags and give feedback
        inputs=None,             # No inputs needed for cancel handler
        outputs=None            # No direct output from cancel handler
        # Removed cancels=[gen_event, batch_event]
    )

    # Connect Video Upload to Audio Extraction and Update Audio Input
    video_audio_input.upload(
        fn=handle_video_upload,
        inputs=[video_audio_input],
        outputs=[audio_input] # Update the main audio input
    )

    # Open Output Folder Button
    open_folder_btn.click(fn=open_folder, inputs=None, outputs=None)

def get_available_drives():
    """Detect available drives on the system regardless of OS"""
    available_paths = []
    if platform.system() == "Windows":
        import string
        from ctypes import windll
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1: drives.append(f"{letter}:\\")
            bitmask >>= 1
        available_paths = drives
    elif platform.system() == "Darwin":
         available_paths = ["/", "/Volumes"]
    else:
        available_paths = ["/", "/mnt", "/media"]

    existing_paths = [p for p in available_paths if os.path.exists(p)]
    print(f"Allowed Gradio paths: {existing_paths}")
    return existing_paths

# --- Launch ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    args, unknown = parser.parse_known_args()

    share_flag = args.share

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True) # Create temp audio dir
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Temporary audio directory: {TEMP_AUDIO_DIR.resolve()}") # Log temp dir

    demo.launch(inbrowser=True, share=share_flag,allowed_paths=get_available_drives()) # Added server_name for listen

# --- END OF FILE secourses_app.py ---