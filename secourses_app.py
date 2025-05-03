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
import re # <<< Added for prompt parsing

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

DEFAULT_PROMPT = "A person speaking animatedly, using expressive hand gestures and nodding their head, medium shot."
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_FPS = 23
DEFAULT_DURATION = 5
MAX_DURATION = 60 # Keep max duration reasonable
DEFAULT_SEED = 500356638
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
    "32GB GPUs": "8,000,000,000",
    "48GB GPUs": "22,000,000,000",
    "80GB GPUs": "32,000,000,000",
}
VRAM_PRESET_DEFAULT = "24GB GPUs" # Adjusted default based on new names

TORCH_DTYPES_STR = {
    "BF16": torch.bfloat16,
    "FP8": torch.float8_e4m3fn,
}
TORCH_DTYPE_DEFAULT = "BF16"
PRESET_DIR = Path("./presets") # Directory to store preset JSON files
# ---------------


# --- Preset Helper Functions ---
def get_preset_files():
    """Scans the PRESET_DIR for .json files and returns a list of preset names (stems)."""
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    presets = sorted([p.stem for p in PRESET_DIR.glob("*.json")])
    return presets if presets else ["Default"] # Ensure "Default" exists if no files found

# --- NEW ---
LAST_PRESET_FILE = PRESET_DIR / "last_preset.txt"

def load_last_used_preset():
    """Loads the name of the last used preset from last_preset.txt."""
    if LAST_PRESET_FILE.exists():
        try:
            last_preset_name = LAST_PRESET_FILE.read_text(encoding='utf-8').strip()
            # Validate that the preset file actually exists
            available_presets = get_preset_files()
            if last_preset_name and last_preset_name in available_presets:
                print(f"[Presets] Loaded last used preset name: {last_preset_name}")
                return last_preset_name
            else:
                print(f"[Presets] Last used preset '{last_preset_name}' not found or invalid. Falling back to 'Default'.")
        except Exception as e:
            print(f"[Error][Presets] Failed to read last used preset file: {e}. Falling back to 'Default'.")
    # Fallback if file doesn't exist or reading fails
    return "Default"

def save_last_used_preset(preset_name):
    """Saves the name of the last used preset to last_preset.txt."""
    try:
        LAST_PRESET_FILE.write_text(preset_name, encoding='utf-8')
        print(f"[Presets] Saved '{preset_name}' as last used preset.")
    except Exception as e:
        print(f"[Error][Presets] Failed to save last used preset '{preset_name}': {e}")
        gr.Warning("Failed to save the last used preset setting.")

# --- REMOVED get_latest_preset_name ---


def get_default_settings():
    """Returns a dictionary containing the default values for all settings."""
    # Map setting name (string) to default value
    return {
        # Inputs / Basic
        "prompt_input": DEFAULT_PROMPT,
        "enable_multi_line_prompts_checkbox": False, # <<< Added default
        "negative_prompt_input": DEFAULT_NEGATIVE_PROMPT,
        "torch_dtype_dropdown": TORCH_DTYPE_DEFAULT,
        "tiled_vae_checkbox": True,
        "width_input": DEFAULT_WIDTH,
        "height_input": DEFAULT_HEIGHT,
        "duration_input": DEFAULT_DURATION,
        "fps_input": DEFAULT_FPS,
        "num_generations_input": 1,
        # CFG / Steps / Seed
        "prompt_cfg_scale": DEFAULT_PROMPT_CFG,
        "audio_cfg_scale": DEFAULT_AUDIO_CFG,
        "audio_weight": DEFAULT_AUDIO_WEIGHT,
        "inference_steps": DEFAULT_INFERENCE_STEPS,
        "seed_input": DEFAULT_SEED,
        "random_seed_checkbox": False,
        # Advanced
        "sigma_shift": DEFAULT_SIGMA_SHIFT,
        "denoising_strength": DEFAULT_DENOISING_STRENGTH,
        "save_video_quality": DEFAULT_SAVE_QUALITY,
        "save_metadata_checkbox": True,
        # Performance / VRAM
        "vram_preset_dropdown": VRAM_PRESET_DEFAULT, # Save the preset name
        "vram_custom_value_input": VRAM_PRESETS[VRAM_PRESET_DEFAULT], # Save the derived value
        "tile_size_h_input": DEFAULT_TILE_SIZE_H,
        "tile_size_w_input": DEFAULT_TILE_SIZE_W,
        "tile_stride_h_input": DEFAULT_TILE_STRIDE_H,
        "tile_stride_w_input": DEFAULT_TILE_STRIDE_W,
        # Batch settings (optional to save, but might be useful)
        # "batch_input_folder_input": "",
        # "batch_output_folder_input": str(OUTPUT_DIR),
        # "batch_skip_existing_checkbox": True,
        # "batch_use_gradio_audio_checkbox": True,
        # "batch_use_gradio_prompt_checkbox": True,
        # <<< RIFE Defaults >>>
        "rife_mode_radio": "None",
        "rife_max_fps_input": 60,
    }

def create_default_preset_if_missing():
    """Creates the presets folder and a 'Default.json' if it doesn't exist."""
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    default_preset_path = PRESET_DIR / "Default.json"
    if not default_preset_path.exists():
        print("[Presets] Default preset not found. Creating...")
        default_settings = get_default_settings()
        try:
            with open(default_preset_path, 'w', encoding='utf-8') as f:
                json.dump(default_settings, f, indent=4)
            print(f"[Presets] Default preset saved to {default_preset_path}")
        except Exception as e:
            print(f"[Error][Presets] Failed to save default preset: {e}")
            gr.Warning("Failed to create default preset file.")

# --- End Preset Helper Functions ---


# --- Core Preset Save/Load Functions ---
# Global list to hold the variable names of components to save/load
# Needs to be populated AFTER the UI components are defined.
SETTING_COMPONENTS_VARS = []

# --- Define the list of setting variable names globally ---
SETTING_COMPONENTS_VARS = [
    # Left Column
    "prompt_input",
    "enable_multi_line_prompts_checkbox", # <<< Added
    "negative_prompt_input",
    "torch_dtype_dropdown", "tiled_vae_checkbox",
    "width_input", "height_input", "duration_input", "fps_input", "num_generations_input",
    "prompt_cfg_scale", "audio_cfg_scale", "audio_weight",
    "inference_steps", "seed_input", "random_seed_checkbox",
    "sigma_shift", "denoising_strength", "save_video_quality", "save_metadata_checkbox",
    "vram_preset_dropdown", "vram_custom_value_input", # Include both VRAM controls
    "tile_size_h_input", "tile_size_w_input", "tile_stride_h_input", "tile_stride_w_input",
    # Batch Tab (Add these if you want presets to save/load batch settings)
    # "batch_input_folder_input", "batch_output_folder_input",
    # "batch_skip_existing_checkbox", "batch_use_gradio_audio_checkbox", "batch_use_gradio_prompt_checkbox",
    # <<< RIFE Settings >>>
    "rife_mode_radio", "rife_max_fps_input",
]

def save_preset(preset_name, *component_values):
    """Saves the current settings to a JSON file."""
    if not preset_name:
        gr.Warning("Please enter a name for the preset.")
        # Return an unchanged dropdown
        return gr.update()

    print(f"[Presets] Attempting to save preset: {preset_name}")
    preset_path = PRESET_DIR / f"{preset_name}.json"

    # Make sure the number of values matches the expected number of components
    if len(component_values) != len(SETTING_COMPONENTS_VARS):
        print(f"[Error][Presets] Mismatch between values provided ({len(component_values)}) and settings expected ({len(SETTING_COMPONENTS_VARS)}). Cannot save.")
        gr.Error("Internal error: Settings count mismatch. Cannot save preset.")
        return gr.Dropdown.update() # Return unchanged dropdown

    # Create the dictionary mapping variable names to current values
    settings_to_save = {}
    for i, var_name in enumerate(SETTING_COMPONENTS_VARS):
        settings_to_save[var_name] = component_values[i]

    try:
        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(settings_to_save, f, indent=4)
        print(f"[Presets] Preset '{preset_name}' saved successfully to {preset_path}")
        # --- ADDED ---
        save_last_used_preset(preset_name) # Update last used preset on save
        # -----------
        gr.Info(f"Preset '{preset_name}' saved.")

        # Refresh the dropdown list and select the newly saved preset
        updated_choices = get_preset_files()
        return gr.update(choices=updated_choices, value=preset_name)

    except Exception as e:
        print(f"[Error][Presets] Failed to save preset '{preset_name}': {e}")
        traceback.print_exc()
        gr.Error(f"Failed to save preset '{preset_name}': {e}")
        return gr.Dropdown.update() # Return unchanged dropdown

def load_preset(preset_name):
    """Loads settings from a preset JSON file and returns updates for UI components."""
    print(f"[Presets] Attempting to load preset: {preset_name}")
    preset_path = PRESET_DIR / f"{preset_name}.json"

    if not preset_path.exists():
        print(f"[Error][Presets] Preset file not found: {preset_path}")
        gr.Error(f"Preset '{preset_name}' not found.")
        # Return updates that do nothing (empty list or list of Nones?)
        # Let's return Nones, assuming len(SETTING_COMPONENTS_VARS) is correct
        # <<< Ensure the returned list has the correct length >>>
        return [None] * len(SETTING_COMPONENTS_VARS) # Must return list of correct length

    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            loaded_settings = json.load(f)
        print(f"[Presets] Preset '{preset_name}' loaded successfully.")
        # --- ADDED ---
        save_last_used_preset(preset_name) # Update last used preset on load
        # -----------

        updates = []
        for var_name in SETTING_COMPONENTS_VARS:
            # <<< Use get with default to handle missing keys gracefully >>>
            setting_value = loaded_settings.get(var_name)
            if setting_value is not None:
                 # Special case: handle boolean default potentially missing from old presets
                if var_name == "enable_multi_line_prompts_checkbox" and var_name not in loaded_settings:
                    print(f"[Warning][Presets] Setting '{var_name}' not found in preset '{preset_name}'. Using default False.")
                    updates.append(gr.update(value=False)) # Default to False if missing
                # <<< Handle new RIFE settings missing from old presets >>>
                elif var_name == "rife_mode_radio" and var_name not in loaded_settings:
                    print(f"[Warning][Presets] Setting '{var_name}' not found in preset '{preset_name}'. Using default 'None'.")
                    updates.append(gr.update(value="None")) # Default to "None" if missing
                elif var_name == "rife_max_fps_input" and var_name not in loaded_settings:
                    print(f"[Warning][Presets] Setting '{var_name}' not found in preset '{preset_name}'. Using default 60.")
                    updates.append(gr.update(value=60)) 
                else:
                    updates.append(gr.update(value=setting_value))
            else:
                # If a setting from the current UI is missing in the preset file,
                # don't update it (keep its current value), except for the new checkbox handled above.
                # <<< Update check to include RIFE handled above >>>
                if var_name not in ["enable_multi_line_prompts_checkbox", "rife_mode_radio", "rife_max_fps_input"]:
                     print(f"[Warning][Presets] Setting '{var_name}' not found in preset '{preset_name}'. Keeping current value.")
                updates.append(gr.update()) # Send an empty update

        # Special handling for VRAM dropdown -> textbox link
        # Find the index of the vram_preset_dropdown and vram_custom_value_input
        try:
            vram_preset_idx = SETTING_COMPONENTS_VARS.index("vram_preset_dropdown")
            vram_custom_idx = SETTING_COMPONENTS_VARS.index("vram_custom_value_input")

            # Get the loaded preset name
            loaded_vram_preset_name = loaded_settings.get("vram_preset_dropdown")

            # If the preset name exists, update the custom textbox accordingly
            if loaded_vram_preset_name in VRAM_PRESETS:
                 expected_custom_value = VRAM_PRESETS[loaded_vram_preset_name]
                 # Update the custom value update object in the list
                 updates[vram_custom_idx] = gr.update(value=expected_custom_value)
                 print(f"[Presets] Updated VRAM custom value based on loaded preset '{loaded_vram_preset_name}'")
            else:
                 # If the loaded preset name is invalid, maybe keep the existing custom value?
                 print(f"[Warning][Presets] Loaded VRAM preset name '{loaded_vram_preset_name}' not found in VRAM_PRESETS. Custom value might be incorrect.")
                 # We already added an update for vram_custom_value_input based on the file,
                 # so we might leave it as is, or force it to None/empty?
                 # For now, leave the value loaded from the file, even if potentially inconsistent.

        except ValueError:
            print("[Warning][Presets] Could not find VRAM dropdown/textbox indices. Skipping VRAM link update.")

        gr.Info(f"Preset '{preset_name}' loaded.")
        return updates

    except Exception as e:
        print(f"[Error][Presets] Failed to load preset '{preset_name}': {e}")
        traceback.print_exc()
        gr.Error(f"Failed to load preset '{preset_name}': {e}")
        # Return updates that do nothing
        return [None] * len(SETTING_COMPONENTS_VARS)

# --- End Core Preset Save/Load Functions ---

# <<< Added Prompt Parsing Helper >>>
def parse_prompts(text: str, min_len: int = 2) -> list[str]:
    """Splits text by newline, trims, filters by length, returns list or default."""
    if not text:
        return [DEFAULT_PROMPT]
    lines = text.splitlines()
    parsed = [line.strip() for line in lines]
    filtered = [line for line in parsed if len(line) >= min_len]
    if not filtered:
        print(f"[Prompt Parsing] No valid prompts found after filtering (min_len={min_len}). Using default prompt.")
        return [DEFAULT_PROMPT]
    # print(f"[Prompt Parsing] Parsed {len(filtered)} prompts: {filtered}") # Debug
    return filtered
# <<< End Added >>>

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
    enable_multi_line_prompts, # <<< Added
    negative_prompt,
    # Basic Settings
    width,
    height,
    duration_seconds,
    fps,
    num_generations, # Number of variations per prompt
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
    # <<< RIFE Settings >>>
    rife_mode,
    rife_max_fps,
    progress=gr.Progress()
):
    """Handles the generation process for single image or multiple sequential generations."""
    global pipe, fantasytalking, wav2vec_processor, wav2vec, models_loaded
    global current_torch_dtype, current_num_persistent_param_in_dit
    global is_generating, is_cancelling, cancel_requested

    print(f"\n>>> generate_video called. Multi-line enabled: {enable_multi_line_prompts}. Current state: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")

    # --- State Check ---
    if is_generating: print("[State Check] Generation already in progress."); gr.Info("A generation task is already running."); return None
    if is_cancelling: print("[State Check] Cancellation in progress."); gr.Info("Cancellation is in progress."); return None

    # --- Set State ---
    is_generating = True; cancel_requested = False
    print(f"[State Check] Set flags at start: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")

    output_video_path = None

    try:
        # --- Input Validation ---
        progress(0, desc="Validating inputs...")
        print("[Validation] Validating inputs...")
        if image_path is None: raise gr.Error("Input Image is required.")
        if audio_path is None: raise gr.Error("Input Audio is required.")

        # (Validation for width, height, duration, fps, num_generations, seed remains the same)
        try: width = int(width); height = int(height)
        except: width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
        if width <= 0 or width % 16 != 0: width = DEFAULT_WIDTH; gr.Warning(f"Invalid width, using default {DEFAULT_WIDTH}")
        if height <= 0 or height % 16 != 0: height = DEFAULT_HEIGHT; gr.Warning(f"Invalid height, using default {DEFAULT_HEIGHT}")
        try: duration_seconds = float(duration_seconds); duration_seconds = max(1, min(duration_seconds, MAX_DURATION))
        except: duration_seconds = DEFAULT_DURATION; gr.Warning("Invalid duration, using default")
        try: fps = int(fps); fps = max(1, fps)
        except: fps = DEFAULT_FPS; gr.Warning("Invalid FPS, using default")
        try: num_generations = int(num_generations); num_generations = max(1, num_generations)
        except: num_generations = 1; gr.Warning("Invalid num generations, using 1")
        initial_seed = DEFAULT_SEED
        try: initial_seed = int(seed); initial_seed = max(0, initial_seed)
        except: initial_seed = DEFAULT_SEED; gr.Warning("Invalid seed, using default")
        print(f"[Validation] Inputs validated. Variations per prompt={num_generations}, Use Random Seed={use_random_seed}, Initial Seed={initial_seed}")

        # --- Audio Handling (Removed complex extraction, assuming direct audio path) ---
        audio_path_to_use = audio_path

        # --- Calculate Target Duration & Frames ---
        progress(0.1, desc="Calculating duration...")
        target_duration = duration_seconds
        try:
            actual_audio_duration = librosa.get_duration(filename=audio_path_to_use)
            print(f"[Audio Info] Effective audio duration: {actual_audio_duration:.2f}s")
            if actual_audio_duration < duration_seconds:
                gr.Warning(f"Requested duration ({duration_seconds}s) > audio ({actual_audio_duration:.2f}s). Using audio duration.")
                target_duration = actual_audio_duration
            if actual_audio_duration <= 0: raise ValueError("Effective audio duration <= 0.")
        except Exception as e: raise gr.Error(f"Could not read audio data from '{Path(audio_path_to_use).name}'. Error: {e}")

        num_frames = calculate_frames(target_duration, fps)
        print(f"[Calculation] Target duration: {target_duration:.2f}s, FPS: {fps}, Calculated num_frames: {num_frames}")
        if num_frames <= 1: raise gr.Error(f"Calculated frames for '{Path(audio_path_to_use).name}' too low ({num_frames}). Check audio/FPS.")

        # --- Prepare Paths and Folders ---
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True); print(f"[File System] Ensured output directory: {OUTPUT_DIR}")
        try: image_path_abs = Path(image_path).resolve().as_posix(); audio_path_abs = Path(audio_path_to_use).resolve().as_posix()
        except Exception as e: raise gr.Error(f"Invalid input file path. Error: {e}")

        # --- Model Loading / Reloading ---
        progress(0.15, desc="Checking model status...") # Moved slightly earlier
        num_persistent_param_in_dit = parse_persistent_params(vram_custom_value_input_param)
        torch_dtype = get_torch_dtype(torch_dtype_str)

        load_needed = False
        if not models_loaded: load_needed = True; print("[Model Check] Models not loaded yet.")
        elif current_torch_dtype != torch_dtype: load_needed = True; print(f"[Model Check] DType changed. Reloading.")
        elif current_num_persistent_param_in_dit != num_persistent_param_in_dit: load_needed = True; print(f"[Model Check] VRAM persistence changed. Reloading.")

        if load_needed:
            print("[Model Loading] Unloading/Reloading models...")
            pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None; models_loaded = False; torch.cuda.empty_cache()
            progress(0.2, desc="Loading models (can take time)...") # Adjusted progress timing
            try:
                pipe, fantasytalking, wav2vec_processor, wav2vec = load_models(MODEL_DIRS["wan_model_dir"], MODEL_DIRS["fantasytalking_model_path"], MODEL_DIRS["wav2vec_model_dir"], num_persistent_param_in_dit, torch_dtype, "cuda")
                models_loaded = True; current_torch_dtype = torch_dtype; current_num_persistent_param_in_dit = num_persistent_param_in_dit
                print("[Model Loading] Models loaded successfully.")
            except Exception as e: print(f"[Error] CRITICAL ERROR loading models: {e}"); traceback.print_exc(); raise gr.Error(f"Failed to load models. Error: {e}")
        else: print("[Model Check] Models ready.")

        # --- <<< Prompt Parsing >>> ---
        if enable_multi_line_prompts:
            prompt_lines = parse_prompts(prompt)
            print(f"[Prompt Info] Multi-line enabled. Parsed {len(prompt_lines)} prompt(s).")
        else:
            prompt_lines = [prompt if prompt else DEFAULT_PROMPT]
            print("[Prompt Info] Multi-line disabled. Using single prompt.")
        num_prompts = len(prompt_lines)

        # --- Generation Loops ---
        total_iterations = num_prompts * num_generations
        current_iteration = 0
        print(f"[Generation] Starting generation loops for {num_prompts} prompt(s) x {num_generations} variation(s) = {total_iterations} total video(s)...")
        cancel_fn = lambda: cancel_requested

        # --- <<< ADDED: Calculate Sequential Base Name (if not batch/random variations) >>> ---
        # Calculate only if num_generations > 1 OR num_prompts > 1, AND not using random seeds,
        # to avoid unnecessary calculation and potential conflicts with truly random seeds giving same number.
        # If it's just one generation (1 prompt, 1 variation), infer.py can handle the sequential naming itself.
        sequential_base_name = None
        # Simplified condition: Calculate base name if we are generating more than one output video
        # for this *single* call to generate_video.
        if total_iterations > 1:
            print(f"[Sequential Naming] Calculating base name for {total_iterations} variations...")
            max_sequence_num = -1 # Start checking from 0000
            # Regex to find filenames starting with 4 digits, optionally followed by _promptX or _variationX
            # We look for the base 4 digits only.
            sequence_pattern = re.compile(r"^(\d{4}).*?\.(mp4|txt)$")
            try:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                for filename in os.listdir(OUTPUT_DIR):
                    match = sequence_pattern.match(filename)
                    if match:
                        num = int(match.group(1))
                        if num > max_sequence_num:
                            max_sequence_num = num
                print(f"[Sequential Naming] Highest sequence number found: {max_sequence_num if max_sequence_num >= 0 else 'None'}")
            except Exception as e:
                print(f"[Warning][Sequential Naming] Error scanning output directory: {e}. Starting sequence from 0.")
                max_sequence_num = -1 # Default to -1 on error, so next is 0

            next_sequence_num = max_sequence_num + 1
            sequential_base_name = f"{next_sequence_num:04d}"
            print(f"[Sequential Naming] Using base sequence number '{sequential_base_name}' for this run.")
        else:
            print(f"[Sequential Naming] Single generation requested ({total_iterations} total). `infer.py` will handle sequential naming if needed.")
        # <<< END ADDED >>>

        # Outer loop: Prompts
        for p_idx, current_prompt in enumerate(prompt_lines):
            prompt_index_for_args = p_idx + 1 if enable_multi_line_prompts and num_prompts > 1 else None # 1-based index or None

            print(f"\n--- Processing Prompt {p_idx + 1}/{num_prompts} ---")
            if enable_multi_line_prompts: print(f"Prompt: '{current_prompt}'")

            # Inner loop: Variations (Number of Generations)
            for i in range(num_generations):
                current_gen_index = i + 1 # Variation index (1-based)
                current_iteration += 1

                progress_desc = f"Prompt {p_idx + 1}/{num_prompts}, Variation {current_gen_index}/{num_generations} ({current_iteration}/{total_iterations})"
                progress(current_iteration / total_iterations * 0.7 + 0.25, desc=progress_desc) # Scale progress: 0.25 to 0.95

                # --- Cancellation Check ---
                if cancel_requested:
                    print(f"[Cancellation] Cancellation detected before Prompt {p_idx+1}, Variation {current_gen_index}.")
                    gr.Warning("Generation cancelled by user.")
                    # Set flag to break outer loop as well
                    cancel_requested = True # Signal outer loop to break
                    break # Exit inner (variation) loop

                # --- Determine Seed ---
                current_seed = 0
                if use_random_seed:
                    current_seed = random.randint(0, 2**32 - 1)
                else:
                    # Seed depends on both prompt index and variation index if multiple prompts/variations
                    current_seed = initial_seed + p_idx * num_generations + i
                print(f"[Seed Info][Prompt {p_idx+1}][Var {current_gen_index}] Using seed: {current_seed}")

                # --- Prepare Arguments for infer.main ---
                print(f"[Args Prep][Prompt {p_idx+1}][Var {current_gen_index}] Preparing arguments for infer.main...")
                args_dict = {
                    "image_path": image_path_abs,
                    "audio_path": audio_path_abs,
                    "prompt": current_prompt, # <<< Use the current prompt line
                    "negative_prompt": negative_prompt,
                    "output_dir": str(OUTPUT_DIR),
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
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
                    # Control info for logging/naming in infer.py
                    "generation_index": current_gen_index, # Variation index (1-based)
                    "total_generations": num_generations, # Total variations for *this* prompt
                    "prompt_index": prompt_index_for_args, # <<< Pass prompt index (or None)
                    "output_base_name": sequential_base_name, # <<< Pass calculated base name (or None if single gen)
                    # Add original FPS to args_dict for RIFE calculation later
                    "original_fps": fps,
                    # <<< Add RIFE settings >>>
                    "rife_mode": rife_mode,
                    "rife_max_fps": rife_max_fps,
                }

                # --- Execute Generation ---
                exec_log_prefix = f"[Exec][P{p_idx+1}][V{current_gen_index}]"
                print(f"{exec_log_prefix} Calling infer.main...")
                try:
                    output_video_path = main(args_dict, pipe, fantasytalking, wav2vec_processor, wav2vec, cancel_fn=cancel_fn, gradio_progress=progress)
                    print(f"{exec_log_prefix} infer.main completed. Output: {output_video_path}")
                    # Update progress *after* successful generation inside loop
                    # progress calculation is already handled before calling main

                except CancelledError as ce:
                    print(f"{exec_log_prefix} Caught CancelledError from infer.main: {ce}")
                    gr.Warning(f"Generation cancelled by user during Prompt {p_idx+1}, Variation {current_gen_index}.")
                    if pipe is not None and hasattr(pipe, 'load_models_to_device'):
                        try: print("[Cancellation] Unloading models..."); pipe.load_models_to_device([]); print("[Cancellation] Models unloaded.")
                        except Exception as unload_e: print(f"[Warning] Error unloading on cancel: {unload_e}")
                    cancel_requested = True # Signal outer loop break
                    break # Break inner variation loop

                except Exception as e:
                    print(f"{exec_log_prefix} Error during infer.main: {e}"); traceback.print_exc()
                    torch.cuda.empty_cache(); print("[Error] CUDA cache clear attempted.")
                    raise gr.Error(f"Error during Prompt {p_idx+1}, Variation {current_gen_index}: {e}")

            # --- End of Inner (Variation) Loop ---
            if cancel_requested:
                break # Break outer (prompt) loop if cancelled

        # --- Loops Finished ---
        if not cancel_requested:
             progress(1.0, desc="All generations complete!")
             print(f"[Generation] All {total_iterations} generation tasks finished.")
        else:
             print("[Generation] Generation loops exited due to cancellation.")

        is_generating = False; is_cancelling = False; cancel_requested = False # Reset all flags
        print(f"[State Check] Exiting FINALLY block for generate_video. Flags after reset: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
        # Optional: torch.cuda.empty_cache()

    except gr.Error as gr_e: print(f"[Gradio Error] {gr_e}"); return None
    except Exception as e: print(f"[Error] Unexpected error in generate_video: {e}"); traceback.print_exc(); gr.Error(f"Unexpected error: {e}"); return None
    finally:
        print(f"[State Check] Entering FINALLY block for generate_video. Current state: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
        is_generating = False; is_cancelling = False; cancel_requested = False # Reset all flags
        print(f"[State Check] Exiting FINALLY block for generate_video. Flags after reset: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
        # Optional: torch.cuda.empty_cache()

    # <<< Explicitly return the last successful video path >>>
    return output_video_path

# --- Function to Handle Video Upload and Audio Extraction ---
# (No changes needed in handle_video_upload)
def handle_video_upload(video_file_path, progress=gr.Progress()):
    """Extracts audio from video, saves to temp, returns path for gr.Audio update."""
    if video_file_path is None:
        print("[Video Handler] No video file provided.")
        return gr.Audio(value=None)

    progress(0, desc="Extracting audio from video...")
    print(f"[Video Handler] Received video file: {video_file_path}")
    video_path_obj = Path(video_file_path)
    extracted_audio_path = None

    # Ensure temp dir exists
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Generate a unique temporary filename for the extracted audio
    temp_audio_filename = f"extracted_audio_{uuid.uuid4()}.wav"
    temp_audio_file = TEMP_AUDIO_DIR / temp_audio_filename

    # Construct the ffmpeg command (force mono WAV)
    ffmpeg_command = [
        'ffmpeg',
        '-i', str(video_path_obj.resolve()),
        '-vn', # No video
        '-acodec', 'pcm_s16le', # Standard WAV codec
        '-ar', '44100', # Sample rate (adjust if needed)
        '-ac', '1', # Mono
        '-y', # Overwrite without asking
        str(temp_audio_file.resolve())
    ]

    print(f"[Video Handler] Running ffmpeg command: {' '.join(ffmpeg_command)}")
    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8', timeout=60) # Added encoding
        # print(f"[Video Handler] FFmpeg stdout: {process.stdout}") # Can be verbose
        # print(f"[Video Handler] FFmpeg stderr: {process.stderr}") # Often has useful info
        if not temp_audio_file.exists() or temp_audio_file.stat().st_size == 0:
             raise RuntimeError("FFmpeg finished but output file is missing or empty.")

        extracted_audio_path = str(temp_audio_file.resolve())
        print(f"[Video Handler] Audio successfully extracted to: {extracted_audio_path}")
        progress(1, desc="Audio extracted!")
        gr.Info(f"Audio extracted from {video_path_obj.name} and loaded.")
        # Return the path to update the gr.Audio component
        return gr.Audio(value=extracted_audio_path, label=f"Extracted Audio from {video_path_obj.name}") # Update label too

    except subprocess.TimeoutExpired:
        print(f"[Error][Video Handler] FFmpeg command timed out after 60 seconds.")
        gr.Warning("Audio extraction took too long and was cancelled.")
        if temp_audio_file.exists(): temp_audio_file.unlink()
        return gr.Audio(value=None) # Clear audio input
    except subprocess.CalledProcessError as e:
        print(f"[Error][Video Handler] FFmpeg failed with exit code {e.returncode}")
        print(f"[Error][Video Handler] FFmpeg stderr: {e.stderr}")
        gr.Error(f"Failed to extract audio from {video_path_obj.name}. Error: {e.stderr[:500]}...")
        return gr.Audio(value=None) # Clear audio input
    except Exception as e:
         print(f"[Error][Video Handler] An unexpected error occurred during audio extraction: {e}")
         traceback.print_exc()
         gr.Error(f"Failed to process video {video_path_obj.name}. Error: {e}")
         return gr.Audio(value=None) # Clear audio input


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
    enable_multi_line_prompts, # <<< Added
    negative_prompt,
    # Basic Settings
    width,
    height,
    duration_seconds,
    fps,
    num_generations_input, # Num variations per prompt per image
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
    # <<< RIFE Settings >>>
    rife_mode,
    rife_max_fps,
    progress=gr.Progress()
):
    """Handles the batch processing of images from a folder."""
    global pipe, fantasytalking, wav2vec_processor, wav2vec, models_loaded
    global current_torch_dtype, current_num_persistent_param_in_dit
    global is_generating, is_cancelling, cancel_requested

    print(f"\n>>> process_batch called. Multi-line enabled: {enable_multi_line_prompts}. Current state: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")

    # --- State Check ---
    if is_generating: print("[State Check][Batch] Generation already in progress."); gr.Info("A generation task is already running."); return None
    if is_cancelling: print("[State Check][Batch] Cancellation in progress."); gr.Info("Cancellation is in progress."); return None

    # --- Set State ---
    is_generating = True; cancel_requested = False
    print(f"[State Check][Batch] Set flags at start: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")

    processed_files_count = 0
    skipped_files_count = 0
    error_files_count = 0
    last_output_path = None

    try:
        # --- Validate Batch Inputs ---
        progress(0, desc="Validating batch inputs...")
        print("[Validation][Batch] Validating batch inputs...")
        if not batch_input_folder_str or not batch_output_folder_str: raise gr.Error("Batch Input and Output folders must be specified.")
        batch_input_folder = Path(batch_input_folder_str)
        batch_output_folder = Path(batch_output_folder_str)
        if not batch_input_folder.is_dir(): raise gr.Error(f"Batch Input folder not found: {batch_input_folder}")
        batch_output_folder.mkdir(parents=True, exist_ok=True); print(f"[File System][Batch] Ensured batch output directory: {batch_output_folder}")

        # --- Find Image Files ---
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}; print(f"[File System][Batch] Scanning for images in: {batch_input_folder}")
        image_files = sorted([p for p in batch_input_folder.glob("*") if p.suffix.lower() in image_extensions])
        if not image_files: raise gr.Error(f"No images found in: {batch_input_folder}")
        total_files = len(image_files); print(f"[File System][Batch] Found {total_files} image file(s).")

        # --- Validate other inputs (re-using validation from generate_video) ---
        try: width = int(width); height = int(height)
        except: width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
        if width <= 0 or width % 16 != 0: width = DEFAULT_WIDTH; gr.Warning(f"[Batch] Invalid width, using default {DEFAULT_WIDTH}")
        if height <= 0 or height % 16 != 0: height = DEFAULT_HEIGHT; gr.Warning(f"[Batch] Invalid height, using default {DEFAULT_HEIGHT}")
        try: duration_seconds = float(duration_seconds); duration_seconds = max(1, min(duration_seconds, MAX_DURATION))
        except: duration_seconds = DEFAULT_DURATION; gr.Warning("[Batch] Invalid duration, using default")
        try: fps = int(fps); fps = max(1, fps)
        except: fps = DEFAULT_FPS; gr.Warning("[Batch] Invalid FPS, using default")
        initial_seed = DEFAULT_SEED
        try: initial_seed = int(seed); initial_seed = max(0, initial_seed)
        except: initial_seed = DEFAULT_SEED; gr.Warning("[Batch] Invalid seed, using default")
        try: num_variations_per_image = int(num_generations_input); num_variations_per_image = max(1, num_variations_per_image)
        except: num_variations_per_image = 1; gr.Warning("[Batch] Invalid num variations, using 1")
        print(f"[Batch Config] Generating {num_variations_per_image} variation(s) per image prompt.")

        # --- Model Loading / Reloading Check (same as single generation) ---
        progress(0.05, desc="Checking model status...") # Slightly earlier
        num_persistent_param_in_dit = parse_persistent_params(vram_custom_value_input_param)
        torch_dtype = get_torch_dtype(torch_dtype_str)

        load_needed = False
        if not models_loaded: load_needed = True; print("[Model Check][Batch] Models not loaded yet.")
        elif current_torch_dtype != torch_dtype: load_needed = True; print(f"[Model Check][Batch] DType changed. Reloading.")
        elif current_num_persistent_param_in_dit != num_persistent_param_in_dit: load_needed = True; print(f"[Model Check][Batch] VRAM persistence changed. Reloading.")

        if load_needed:
            print("[Model Loading][Batch] Unloading/Reloading models...")
            pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None; models_loaded = False; torch.cuda.empty_cache()
            progress(0.1, desc="Loading models for batch...") # Adjusted progress
            try:
                pipe, fantasytalking, wav2vec_processor, wav2vec = load_models(MODEL_DIRS["wan_model_dir"], MODEL_DIRS["fantasytalking_model_path"], MODEL_DIRS["wav2vec_model_dir"], num_persistent_param_in_dit, torch_dtype, "cuda")
                models_loaded = True; current_torch_dtype = torch_dtype; current_num_persistent_param_in_dit = num_persistent_param_in_dit
                print("[Model Loading][Batch] Models loaded.")
            except Exception as e: print(f"[Error][Batch] CRITICAL ERROR loading models: {e}"); traceback.print_exc(); raise gr.Error(f"Failed to load models for batch. Error: {e}")
        else: print("[Model Check][Batch] Models ready.")

        # --- Batch Processing Loops ---
        cancel_fn = lambda: cancel_requested
        # Estimate total iterations (will be refined inside loop)
        estimated_total_iterations = total_files * num_variations_per_image # Initial estimate
        current_iteration = 0
        actual_total_iterations = 0 # Will calculate precisely

        # Pre-calculate total iterations precisely
        for i, image_path in enumerate(image_files):
            image_stem = image_path.stem
            # Determine prompt source text
            potential_prompt_path = batch_input_folder / f"{image_stem}.txt"
            prompt_to_parse = prompt_fallback # Start with fallback
            if potential_prompt_path.exists():
                 try: prompt_to_parse = potential_prompt_path.read_text(encoding='utf-8').strip()
                 except Exception: pass # Keep fallback if read fails
            elif not batch_use_gradio_prompt:
                 prompt_to_parse = DEFAULT_PROMPT # Force default if fallback disabled

            # Parse prompts based on mode
            if enable_multi_line_prompts:
                 prompt_lines = parse_prompts(prompt_to_parse)
            else:
                 prompt_lines = [prompt_to_parse if prompt_to_parse else DEFAULT_PROMPT]
            actual_total_iterations += len(prompt_lines) * num_variations_per_image
        print(f"[Batch][Prep] Calculated total iterations: {actual_total_iterations}")


        print(f"[Batch] Starting batch processing loop for {total_files} image(s)...")
        # Outer loop: Images
        for i, image_path in enumerate(image_files):
            image_stem = image_path.stem
            item_log_prefix = f"[Batch Item {i + 1}/{total_files} ({image_path.name})]"
            print(f"\n--- Processing {item_log_prefix} ---")

            if cancel_requested: print(f"{item_log_prefix} Cancellation detected. Stopping batch."); break

            # --- Find Audio (remains the same) ---
            audio_file_to_use = None; audio_extensions = {".wav", ".mp3", ".flac"}; found_audio = False
            for ext in audio_extensions:
                 potential_audio_path = batch_input_folder / f"{image_stem}{ext}"
                 if potential_audio_path.exists(): audio_file_to_use = potential_audio_path.resolve().as_posix(); found_audio = True; break
            if not found_audio:
                 if batch_use_gradio_audio and audio_input_fallback: audio_file_to_use = Path(audio_input_fallback).resolve().as_posix()
                 else: print(f"{item_log_prefix} No audio found/fallback disabled. Skipping."); error_files_count += 1; continue

            # --- Find and Parse Prompts ---
            potential_prompt_path = batch_input_folder / f"{image_stem}.txt"
            prompt_source_text = prompt_fallback # Start with fallback
            prompt_source_desc = "UI fallback"
            if potential_prompt_path.exists():
                try:
                    prompt_source_text = potential_prompt_path.read_text(encoding='utf-8').strip()
                    prompt_source_desc = f"file ({potential_prompt_path.name})"
                    print(f"{item_log_prefix} Found matching prompt file.")
                except Exception as e:
                    print(f"{item_log_prefix} Warning: Failed to read prompt file {potential_prompt_path.name}: {e}. Using {prompt_source_desc}.")
                    gr.Warning(f"Error reading prompt file {potential_prompt_path.name}")
            elif not batch_use_gradio_prompt:
                 prompt_source_text = DEFAULT_PROMPT # Force default if fallback disabled
                 prompt_source_desc = "default (UI fallback disabled)"

            if enable_multi_line_prompts:
                prompt_lines = parse_prompts(prompt_source_text)
                print(f"{item_log_prefix} Multi-line enabled. Parsed {len(prompt_lines)} prompt(s) from {prompt_source_desc}.")
            else:
                prompt_lines = [prompt_source_text if prompt_source_text else DEFAULT_PROMPT]
                print(f"{item_log_prefix} Multi-line disabled. Using single prompt from {prompt_source_desc}.")
            num_prompts = len(prompt_lines)

            # --- Calculate Duration & Frames (remains the same) ---
            current_target_duration = duration_seconds
            try: item_audio_duration = librosa.get_duration(filename=audio_file_to_use); print(f"{item_log_prefix} Audio duration: {item_audio_duration:.2f}s")
            except Exception as e: print(f"{item_log_prefix} Failed read duration for {Path(audio_file_to_use).name}. Skipping. Error: {e}"); error_files_count += 1; continue
            if item_audio_duration < duration_seconds: current_target_duration = item_audio_duration
            if item_audio_duration <= 0: print(f"{item_log_prefix} Audio duration <= 0. Skipping."); error_files_count += 1; continue
            current_num_frames = calculate_frames(current_target_duration, fps); print(f"{item_log_prefix} Target duration: {current_target_duration:.2f}s, num_frames: {current_num_frames}")
            if current_num_frames <= 1: print(f"{item_log_prefix} Calculated frames <= 1. Skipping."); error_files_count += 1; continue


            # Middle loop: Prompts
            for p_idx, current_prompt in enumerate(prompt_lines):
                prompt_index_for_args = p_idx + 1 if enable_multi_line_prompts and num_prompts > 1 else None
                prompt_log_prefix = f"[P{p_idx + 1}/{num_prompts}]"

                if cancel_requested: print(f"{item_log_prefix}{prompt_log_prefix} Cancellation detected. Stopping."); break # Break middle loop

                # Inner loop: Variations
                for j in range(num_variations_per_image):
                    variation_index = j + 1 # 1-based variation index
                    variation_log_prefix = f"[V{variation_index}/{num_variations_per_image}]"
                    current_iteration += 1

                    progress_desc = f"Img {i+1}/{total_files}, Prompt {p_idx+1}/{num_prompts}, Var {variation_index}/{num_variations_per_image} ({current_iteration}/{actual_total_iterations})"
                    progress(current_iteration / actual_total_iterations * 0.85 + 0.1, desc=progress_desc) # Scale progress: 0.1 to 0.95

                    if cancel_requested: print(f"{item_log_prefix}{prompt_log_prefix}{variation_log_prefix} Cancellation detected. Stopping."); break # Break inner loop

                    # --- Skip Logic (incorporating prompt index) ---
                    prompt_suffix = f"_prompt{prompt_index_for_args}" if prompt_index_for_args is not None else ""
                    variation_suffix = f"_{variation_index:04d}" if num_variations_per_image > 1 else ""
                    output_video_name = f"{image_stem}{prompt_suffix}{variation_suffix}.mp4"
                    output_meta_name = f"{image_stem}{prompt_suffix}{variation_suffix}.txt"
                    output_video_path = batch_output_folder / output_video_name
                    output_metadata_path = batch_output_folder / output_meta_name

                    if batch_skip_existing and (output_video_path.exists() or (save_metadata and output_metadata_path.exists())):
                        print(f"{item_log_prefix}{prompt_log_prefix}{variation_log_prefix} Skipping '{output_video_name}' (already exists).")
                        skipped_files_count += 1
                        continue # Skip to next variation

                    # --- Determine Seed ---
                    current_seed = 0
                    if use_random_seed: current_seed = random.randint(0, 2**32 - 1)
                    else: current_seed = initial_seed + i * num_prompts * num_variations_per_image + p_idx * num_variations_per_image + j
                    print(f"{item_log_prefix}{prompt_log_prefix}{variation_log_prefix} Using seed: {current_seed}")

                    # --- Prepare Arguments for infer.main ---
                    print(f"{item_log_prefix}{prompt_log_prefix}{variation_log_prefix} Preparing args for infer.main...")
                    args_dict = {
                        "image_path": image_path.resolve().as_posix(),
                        "audio_path": audio_file_to_use,
                        "prompt": current_prompt, # <<< Use current prompt line
                        "negative_prompt": negative_prompt,
                        "output_dir": str(batch_output_folder),
                        "width": width, "height": height, "num_frames": current_num_frames, "fps": fps,
                        "audio_weight": float(audio_weight), "prompt_cfg_scale": float(prompt_cfg_scale), "audio_cfg_scale": float(audio_cfg_scale),
                        "inference_steps": int(inference_steps), "seed": current_seed,
                        "tiled_vae": bool(tiled_vae),
                        "tile_size_h": int(tile_size_h), "tile_size_w": int(tile_size_w),
                        "tile_stride_h": int(tile_stride_h), "tile_stride_w": int(tile_stride_w),
                        "sigma_shift": float(sigma_shift), "denoising_strength": float(denoising_strength),
                        "save_video_quality": int(save_video_quality), "save_metadata": bool(save_metadata),
                        # --- Control info for infer.py ---
                        "generation_index": variation_index,        # Variation index (1-based)
                        "total_generations": num_variations_per_image, # Total variations for this prompt/image combo
                        "prompt_index": prompt_index_for_args,      # Prompt index for multi-line (1-based or None)
                        "output_base_name": image_stem,          # Base name for infer.py naming
                        # Add original FPS to args_dict for RIFE calculation later
                        "original_fps": fps,
                        # <<< Add RIFE settings >>>
                        "rife_mode": rife_mode,
                        "rife_max_fps": rife_max_fps,
                    }

                    # --- Execute Generation ---
                    exec_log_prefix = f"[Exec]{item_log_prefix}{prompt_log_prefix}{variation_log_prefix}"
                    try:
                        print(f"{exec_log_prefix} Calling infer.main for '{output_video_name}'...")
                        last_output_path = main(args_dict, pipe, fantasytalking, wav2vec_processor, wav2vec, cancel_fn=cancel_fn, gradio_progress=progress)
                        print(f"{exec_log_prefix} Successfully processed. Output: {last_output_path}")
                        processed_files_count += 1
                    except CancelledError as ce:
                        print(f"{exec_log_prefix} Caught CancelledError: {ce}"); gr.Warning(f"Batch cancelled during {output_video_name}.")
                        cancel_requested = True; break # Break inner loop
                    except Exception as e:
                        print(f"{exec_log_prefix} Failed: {e}"); traceback.print_exc()
                        gr.Warning(f"Error processing {output_video_name}: {e}. Skipping.")
                        error_files_count += 1; continue # Continue to next variation

                # --- End Inner (Variation) Loop ---
                if cancel_requested: break # Break middle loop

            # --- End Middle (Prompt) Loop ---
            if cancel_requested: break # Break outer loop

        # --- Batch Loops Finished ---
        if not cancel_requested:
             final_desc = f"Batch complete! Processed: {processed_files_count}, Skipped: {skipped_files_count}, Errors: {error_files_count} (Total Iterations: {actual_total_iterations})."
             progress(1.0, desc=final_desc)
             print(f"[Batch] Batch finished. {final_desc}")
             gr.Info(final_desc)
        else:
             print("[Batch] Batch processing loop exited due to cancellation.")
             gr.Warning(f"Batch cancelled. Processed: {processed_files_count}, Skipped: {skipped_files_count}, Errors: {error_files_count} before cancel.")

        return last_output_path

    except gr.Error as gr_e: print(f"[Gradio Error][Batch] {gr_e}"); return None
    except Exception as e: print(f"[Error][Batch] Unexpected error: {e}"); traceback.print_exc(); gr.Error(f"Unexpected batch error: {e}"); return None
    finally:
        print(f"[State Check] Entering FINALLY block for process_batch. Current state: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")
        is_generating = False; is_cancelling = False; cancel_requested = False # Reset all flags
        print(f"[State Check] Exiting FINALLY block for process_batch. Flags after reset: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")

        is_generating = False; is_cancelling = False; cancel_requested = False # Reset all flags
        print(f"[State Check] Exiting FINALLY block for process_batch. Flags after reset: is_generating={is_generating}, is_cancelling={is_cancelling}, cancel_requested={cancel_requested}")


# --- Cancel Handler ---
def handle_cancel():
    """Sets the cancellation flag and provides user feedback."""
    global is_generating, is_cancelling, cancel_requested

    if not is_generating or is_cancelling:
        print("[Cancel Handler] No active task or already cancelling.")
        return

    print("[Cancel Handler] Cancel requested. Setting flags.")
    cancel_requested = True
    is_cancelling = True # Prevent new runs & repeated cancels
    # Don't reset is_generating here, the finally block of the running task will do it
    gr.Warning("Cancel requested! Attempting to stop generation...")


# --- Gradio UI Definition ---
with gr.Blocks(title="FantasyTalking Video Generation (SECourses App V15)", theme=gr.themes.Soft()) as demo: # Updated title
    gr.Markdown(
        """
    # FantasyTalking: Realistic Talking Portrait Generation SECourses App V15 - https://www.patreon.com/posts/127855145
    Generate a talking head video from an image and audio, or process a batch of images. Supports multiple prompts per generation.
    [GitHub](https://github.com/Fantasy-AMAP/fantasy-talking) | [arXiv Paper](https://arxiv.org/abs/2504.04842)
    """
    ) # Updated description slightly

    with gr.Row():
        # --- Left Column: Inputs & Basic Settings ---
        with gr.Column(scale=2):
            gr.Markdown("## 1. Inputs (Single Generation)")
            with gr.Row():
                image_input = gr.Image(label="Input Image", type="filepath", scale=1)
                audio_input = gr.Audio(label="Input Audio (WAV/MP3)", type="filepath", scale=1) # Reverted to gr.Audio

            video_audio_input = gr.File(
                label="Or Upload Video to Extract Audio",
                file_types=["video", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"],
                type="filepath",height=150
            )

            prompt_input = gr.Textbox(
                label="Prompt(s)", placeholder=DEFAULT_PROMPT, value=DEFAULT_PROMPT, lines=5,
                info="Enter one prompt per line. Enable checkbox below to process each line separately." # Updated info
            )
            # <<< Added Multi-Line Checkbox >>>
            enable_multi_line_prompts_checkbox = gr.Checkbox(
                label="Enable Multi-Line Prompts",
                value=False, # Default is off
                info="If checked, each line in the Prompt(s) box above (min 2 chars) will generate a separate video sequence."
            )
            # <<< End Added >>>

            negative_prompt_input = gr.Textbox(
                label="Negative Prompt", placeholder=DEFAULT_NEGATIVE_PROMPT, value=DEFAULT_NEGATIVE_PROMPT, lines=2,
                info="Describe what NOT to generate. Applied to all prompts." # Clarified scope
            )
            with gr.Row():
                torch_dtype_dropdown = gr.Dropdown(
                        choices=list(TORCH_DTYPES_STR.keys()), value=TORCH_DTYPE_DEFAULT, label="Model Loading Precision", # Shortened label
                        info="BF16=Quality/RAM/VRAM+, FP8=Quality-/RAM/VRAM-. Needs reload." # Shortened info
                    )
                tiled_vae_checkbox = gr.Checkbox(label="Enable Tiled VAE", value=True, info="Saves VRAM during decode, may be slower.")

            gr.Examples(
                examples=[
                    ["assets/images/man.png", "assets/audios/man.wav", "A person speaking animatedly, using expressive hand gestures and nodding their head, medium shot."] # Example with multi-line
                ],
                inputs=[image_input, audio_input, prompt_input],
                label="Examples (Image, Audio, Prompt(s))" # Updated label
            )

            gr.Markdown("## 2. Generation Settings")
            with gr.Row():
                 width_input = gr.Number(value=DEFAULT_WIDTH, label="Width", minimum=64, step=16, precision=0, info="/ 16") # Shortened info
                 height_input = gr.Number(value=DEFAULT_HEIGHT, label="Height", minimum=64, step=16, precision=0, info="/ 16") # Shortened info
            with gr.Row():
                duration_input = gr.Number(value=DEFAULT_DURATION, minimum=1, maximum=MAX_DURATION, label="Max Duration (s)", info=f"<= audio, max {MAX_DURATION}s") # Shortened info
                fps_input = gr.Number(value=DEFAULT_FPS, minimum=1, maximum=60, label="FPS", precision=0)

            num_generations_input = gr.Number(label="Number of Generations per Prompt", value=1, minimum=1, step=1, precision=0, info="Number of videos per prompt line (uses different seeds).") # Updated label/info

            with gr.Row():
                prompt_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, value=DEFAULT_PROMPT_CFG, step=0.5, label="Prompt CFG")
                audio_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, value=DEFAULT_AUDIO_CFG, step=0.5, label="Audio CFG")
            with gr.Row():
                audio_weight = gr.Slider(minimum=0.1, maximum=3.0, value=DEFAULT_AUDIO_WEIGHT, step=0.1, label="Audio Weight", info="Audio influence.") # Shortened info
                inference_steps = gr.Slider(minimum=1, maximum=100, value=DEFAULT_INFERENCE_STEPS, step=1, label="Inference Steps")

            with gr.Row():
                seed_input = gr.Number(value=DEFAULT_SEED, label="Seed", minimum=-1, precision=0, info="-1 default")
                random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=False, info="Overrides Seed.")

            with gr.Accordion("Advanced Settings", open=True):
                with gr.Row():
                    sigma_shift = gr.Slider(minimum=0.1, maximum=10.0, value=DEFAULT_SIGMA_SHIFT, step=0.1, label="Sigma Shift")
                    denoising_strength = gr.Slider(minimum=0.1, maximum=1.0, value=DEFAULT_DENOISING_STRENGTH, step=0.05, label="Denoising Strength", info="Usually 1.0 for I2V.")
                with gr.Row():
                    save_video_quality = gr.Slider(minimum=0, maximum=51, value=DEFAULT_SAVE_QUALITY, step=1, label="Output Quality (CRF)", info="Lower=Higher quality (0=lossless).") # Simplified info
                    save_metadata_checkbox = gr.Checkbox(label="Save Metadata (.txt)", value=True, info="Save settings, timings.")

            with gr.Accordion("Performance & VRAM", open=True):
                gr.Markdown("_Higher resolution/duration needs more VRAM._")
                with gr.Row():
                    vram_preset_dropdown = gr.Dropdown(choices=list(VRAM_PRESETS.keys()), value=VRAM_PRESET_DEFAULT, label="VRAM Preset Helper")
                    vram_custom_value_input = gr.Textbox(label="Persistent Params Value", value=VRAM_PRESETS[VRAM_PRESET_DEFAULT], info="Params in VRAM. 0=Least VRAM/Slowest. Adjust for resolution/duration.") # Simplified info
                vram_preset_dropdown.change(fn=update_vram_textbox, inputs=vram_preset_dropdown, outputs=vram_custom_value_input)

                with gr.Group(visible=True) as tile_options:
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
                process_btn = gr.Button("Generate Video(s)", variant="primary", scale=2) # Renamed button slightly
                cancel_btn = gr.Button("Cancel All", variant="stop", scale=1)
            video_output = gr.Video(label="Last Generated Video Output") # Clarified label
            open_folder_btn = gr.Button("Open Outputs Folder")

            with gr.Accordion("Batch Processing", open=True):
                 gr.Markdown(
                     """
                     Process images from a folder. Looks for matching audio (`<name>.[wav|mp3|flac]`)
                     and prompts (`<name>.txt`) in the *input folder*. If 'Enable Multi-Line Prompts' is checked,
                     each line in the `.txt` file (or UI fallback prompt) will be processed. Uses UI settings as defaults.
                     """ # Updated description
                 )
                 batch_input_folder_input = gr.Textbox(label="Batch Input Folder", placeholder="/path/to/your/inputs")
                 batch_output_folder_input = gr.Textbox(label="Batch Output Folder", placeholder="/path/to/your/outputs", value=str(OUTPUT_DIR))
                 with gr.Row():
                      batch_skip_existing_checkbox = gr.Checkbox(label="Skip Existing", value=True) # Shortened
                      batch_use_gradio_audio_checkbox = gr.Checkbox(label="Use UI Audio Fallback", value=True) # Shortened
                      batch_use_gradio_prompt_checkbox = gr.Checkbox(label="Use UI Prompt Fallback", value=True) # Shortened

                 batch_start_btn = gr.Button("Start Batch Process", variant="primary")

            # --- Presets Section ---
            with gr.Group():
                gr.Markdown("## 4. Presets")
                initial_presets = get_preset_files()
                initial_preset_name = load_last_used_preset()
                preset_dropdown = gr.Dropdown(choices=initial_presets, value=initial_preset_name, label="Load Preset") # Simplified
                with gr.Row():
                    preset_name_input = gr.Textbox(label="Save Preset As", placeholder="Enter preset name...")
                    save_preset_btn = gr.Button("Save Current Settings")

            # --- RIFE Section (Added above Presets) ---
            with gr.Accordion("3a. RIFE Frame Interpolation (Optional)", open=True):
                gr.Markdown(
                    """
                    Apply RIFE AI (SOTA Frame Increase Model) frame interpolation to increase the video's FPS *after* generation.
                    Output filename will have `_RIFE_2x` or `_RIFE_4x` appended.
                    """
                )
                rife_mode_radio = gr.Radio(
                    ["None", "2x FPS", "4x FPS"],
                    value="None", # Default to None
                    label="RIFE Mode",
                    info="Select FPS multiplication factor. 'None' disables RIFE."
                )
                rife_max_fps_input = gr.Number(
                    value=60, # Default limit
                    label="Max RIFE FPS Limit",
                    minimum=1,
                    step=1,
                    precision=0,
                    info="Limit the final FPS after RIFE. E.g., 23fps * 2x = 46fps. If limit is 60, output will be 60fps."
                )
    # --- Event Handling ---

    # Single Generation Button
    gen_inputs = [
            image_input, audio_input, prompt_input,
            enable_multi_line_prompts_checkbox, # <<< Added
            negative_prompt_input,
            width_input, height_input, duration_input, fps_input, num_generations_input,
            prompt_cfg_scale, audio_cfg_scale, audio_weight,
            inference_steps, seed_input, random_seed_checkbox,
            sigma_shift, denoising_strength, save_video_quality, save_metadata_checkbox,
            tiled_vae_checkbox, tile_size_h_input, tile_size_w_input, tile_stride_h_input, tile_stride_w_input,
            vram_custom_value_input,
            torch_dtype_dropdown,
            rife_mode_radio, rife_max_fps_input,
        ]
    gen_event = process_btn.click(fn=generate_video, inputs=gen_inputs, outputs=video_output)

    # Batch Generation Button
    batch_inputs = [
              batch_input_folder_input, batch_output_folder_input,
              batch_skip_existing_checkbox, batch_use_gradio_audio_checkbox, batch_use_gradio_prompt_checkbox,
              image_input, audio_input, prompt_input,
              enable_multi_line_prompts_checkbox, # <<< Added
              negative_prompt_input,
              width_input, height_input, duration_input, fps_input,
              num_generations_input,
              prompt_cfg_scale, audio_cfg_scale, audio_weight,
              inference_steps, seed_input, random_seed_checkbox,
              sigma_shift, denoising_strength, save_video_quality, save_metadata_checkbox,
              tiled_vae_checkbox, tile_size_h_input, tile_size_w_input, tile_stride_h_input, tile_stride_w_input,
              vram_custom_value_input,
              torch_dtype_dropdown,
              rife_mode_radio, rife_max_fps_input,
         ]
    batch_event = batch_start_btn.click(fn=process_batch, inputs=batch_inputs, outputs=video_output)

    # --- Populate the Component List for Presets ---
    # Ensure order matches SETTING_COMPONENTS_VARS
    SETTING_COMPONENTS = [
        prompt_input,
        enable_multi_line_prompts_checkbox, # <<< Added
        negative_prompt_input,
        torch_dtype_dropdown, tiled_vae_checkbox,
        width_input, height_input, duration_input, fps_input, num_generations_input,
        prompt_cfg_scale, audio_cfg_scale, audio_weight,
        inference_steps, seed_input, random_seed_checkbox,
        sigma_shift, denoising_strength, save_video_quality, save_metadata_checkbox,
        vram_preset_dropdown, vram_custom_value_input,
        tile_size_h_input, tile_size_w_input, tile_stride_h_input, tile_stride_w_input,
        # <<< RIFE Settings >>>
        rife_mode_radio, rife_max_fps_input,
    ]

    # --- Preset Event Handling ---
    save_preset_btn.click(fn=save_preset, inputs=[preset_name_input] + SETTING_COMPONENTS, outputs=[preset_dropdown])
    preset_dropdown.change(fn=load_preset, inputs=[preset_dropdown], outputs=SETTING_COMPONENTS)

    # Universal Cancel Button
    cancel_btn.click(fn=handle_cancel, inputs=None, outputs=None) # Removed cancels= list

    # Connect Video Upload to Audio Extraction and Update Audio Input
    video_audio_input.upload(fn=handle_video_upload, inputs=[video_audio_input], outputs=[audio_input])

    # Open Output Folder Button
    open_folder_btn.click(fn=open_folder, inputs=None, outputs=None)

    # --- Apply Initial Settings on App Load ---
    def apply_initial_settings():
        preset_to_load = load_last_used_preset()
        print(f"[Startup] Applying initial preset: {preset_to_load}")
        try:
            # <<< Ensure list has correct length even on error >>>
            updates = load_preset(preset_to_load)
            if len(updates) != len(SETTING_COMPONENTS):
                 print(f"[Error][Startup] Incorrect number of updates returned by load_preset ({len(updates)} vs {len(SETTING_COMPONENTS)}). Using defaults.")
                 return [None] * len(SETTING_COMPONENTS)
            print(f"[Startup] Successfully applied initial preset: {preset_to_load}")
            return updates
        except Exception as e:
            print(f"[Error][Startup] Failed during initial preset application for '{preset_to_load}': {e}")
            return [None] * len(SETTING_COMPONENTS)

    demo.load(apply_initial_settings, inputs=None, outputs=SETTING_COMPONENTS)

# --- Utility Functions (No changes needed) ---
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

    create_default_preset_if_missing()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Temporary audio directory: {TEMP_AUDIO_DIR.resolve()}")

    demo.launch(inbrowser=True, share=share_flag, allowed_paths=get_available_drives())
