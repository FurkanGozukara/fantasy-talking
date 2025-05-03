# Copyright Alibaba Inc. All Rights Reserved.

import os
import subprocess
from datetime import datetime
from pathlib import Path
import time # For calculating duration
import json # For metadata saving and potentially prompt loading
import pprint # For pretty printing dicts
import traceback # For detailed error logging
import shutil # <<< Added for file copying
import re # <<< Added for regex in sequential naming
import sys # <<< Added for sys.executable in RIFE call

# import cv2 # No longer needed directly here
import librosa # Keep for checking audio duration before processing
import torch
from PIL import Image
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np # Keep for potential future use and video array conversion
from tqdm import tqdm # <<< Add this import

from diffsynth import ModelManager, WanVideoPipeline
from model import FantasyTalkingAudioConditionModel
from utils import get_audio_features, resize_image_by_longest_edge, save_video # Assuming save_video still needed for temp file

# --- Import the custom exception --- #
# Ensure wan_video.py is accessible or the CancelledError is defined here
try:
    # Use the specific exception from the pipeline module if available
    from diffsynth.pipelines.wan_video import CancelledError
except ImportError:
    print("[Warning][infer.py] Could not import CancelledError from diffsynth.pipelines.wan_video. Defining locally.")
    class CancelledError(Exception):
        """Custom exception for cancellation (local definition)."""
        pass
# ---------------------------------- #

# --- Define Used Audios Directory --- #
USED_AUDIO_DIR = Path("./used_audios")
# ----------------------------------- #


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
    print("[Model Loading] Attempting to load models...")
    print(f"[Model Loading] Target Device: {device}")
    print(f"[Model Loading] Target Dtype: {torch_dtype}")
    print(f"[Model Loading] Wan Model Dir: {wan_model_dir}")
    print(f"[Model Loading] FantasyTalking Model Path: {fantasytalking_model_path}")
    print(f"[Model Loading] Wav2Vec Model Dir: {wav2vec_model_dir}")
    # Handle None value for printing
    persistent_params_str = str(num_persistent_param_in_dit) if num_persistent_param_in_dit is not None else "None (Max Persistence)"
    print(f"[Model Loading] Persistent Params (Parsed): {persistent_params_str}")
    # ------------------------

    # Load Wan I2V models
    print("[Model Loading] Loading Wan I2V models to CPU first...")
    model_manager = ModelManager(device="cpu") # Keep on CPU initially

    # --- Determine Wan I2V model file and loading dtype based on pipeline's target dtype ---
    target_pipeline_dtype = torch_dtype # This is the dtype passed from the app
    wan_model_filename = ""
    wan_loading_dtype = None

    if target_pipeline_dtype in [torch.bfloat16, torch.float16]:
        wan_model_filename = "wan21_i2v_720p_14B_fp16.safetensors"
        wan_loading_dtype = torch.bfloat16
        print(f"[Model Loading] Selected Wan I2V model for BF16/FP16 pipeline: {wan_model_filename} (loading as {torch_dtype})")
    elif target_pipeline_dtype == torch.float8_e4m3fn:
        wan_model_filename = "wan21_i2v_720p_14B_fp8_e4m3fn.safetensors"
        wan_loading_dtype = torch_dtype # Load FP8 model as FP8
        print(f"[Model Loading] Selected Wan I2V model for FP8 pipeline: {wan_model_filename} (loading as {torch_dtype})")
    else:
        # Fallback or error - defaulting to FP8 might be risky, but let's match previous behavior slightly safer
        print(f"[Warning][Model Loading] Unsupported torch_dtype ({target_pipeline_dtype}) received. Defaulting to FP8 model. Behavior might be unexpected.")
        wan_model_filename = "wan21_i2v_720p_14B_fp8_e4m3fn.safetensors"
        wan_loading_dtype = torch.bfloat16

    if not wan_model_filename or not wan_loading_dtype:
         # This should ideally not happen with the fallback, but good practice
         raise ValueError(f"Could not determine Wan I2V model file or loading dtype for pipeline dtype: {target_pipeline_dtype}")

    wan_model_path = os.path.join(wan_model_dir, wan_model_filename)
    # ----------------------------------------------------------------------------------

    try:
        model_manager.load_models(
            [
                # Dynamically determined Wan I2V model path
                wan_model_path,
                # Other models remain the same
                os.path.join(wan_model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                os.path.join(wan_model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                os.path.join(wan_model_dir, "Wan2.1_VAE.pth"),
            ],
            torch_dtype=wan_loading_dtype, # Use the determined dtype for loading this set
        )
    except Exception as e:
        print(f"[Error][Model Loading] Failed to load one or more Wan I2V models from {wan_model_dir}. Check paths and files.")
        print(f"[Error][Model Loading] Details: {e}")
        traceback.print_exc()
        raise # Re-raise critical error

    print("[Model Loading] Wan I2V models loaded to CPU.")
    print("[Model Loading] Creating WanVideoPipeline...")
    try:
        # <<< Simplified pipeline creation, dtype handled by from_model_manager logic internally now >>>
        # (Assuming diffsynth >= 0.0.17 where device/dtype handled more directly)
        pipe = WanVideoPipeline.from_model_manager(
            model_manager, torch_dtype=torch.bfloat16, device=device
        )
    except Exception as e:
        print(f"[Error][Model Loading] Failed to create WanVideoPipeline.")
        print(f"[Error][Model Loading] Details: {e}")
        traceback.print_exc()
        raise
    print("[Model Loading] WanVideoPipeline created.")

    # Load FantasyTalking weights
    print("[Model Loading] Loading FantasyTalking weights...")
    try:
        fantasytalking = FantasyTalkingAudioConditionModel(pipe.dit, 768, 2048).to(device)
        fantasytalking.load_audio_processor(fantasytalking_model_path, pipe.dit)
    except FileNotFoundError:
         print(f"[Error][Model Loading] FantasyTalking model file not found at: {fantasytalking_model_path}")
         raise
    except Exception as e:
         print(f"[Error][Model Loading] Failed to load FantasyTalking weights.")
         print(f"[Error][Model Loading] Details: {e}")
         traceback.print_exc()
         raise
    print("[Model Loading] FantasyTalking weights loaded and processor attached.")

    # Enable VRAM management
    try:
        if num_persistent_param_in_dit is not None:
             print(f"[VRAM Management] Enabling VRAM management with num_persistent_param_in_dit: {num_persistent_param_in_dit}")
             pipe.enable_vram_management(
                 num_persistent_param_in_dit=num_persistent_param_in_dit # Pass int directly
             )
        else:
            print("[VRAM Management] VRAM management using max persistence ('None' value used).")
            # Optionally call enable_vram_management without the limit if that's how it enables default behavior
            # pipe.enable_vram_management() # Or however unlimited persistence is set
    except Exception as e:
         print(f"[Error][VRAM Management] Failed during VRAM management setup.")
         print(f"[Error][VRAM Management] Details: {e}")
         # Decide if this is critical - potentially continue without optimal management?
         # raise # Or just log warning

    # Load wav2vec models
    print("[Model Loading] Loading Wav2Vec models...")
    try:
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
        wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_dir).to(device)
    except Exception as e:
         print(f"[Error][Model Loading] Failed to load Wav2Vec models from {wav2vec_model_dir}. Check path and files.")
         print(f"[Error][Model Loading] Details: {e}")
         traceback.print_exc()
         raise
    print("[Model Loading] Wav2Vec models loaded.")
    print("-" * 20)
    print("[Model Loading] All models loaded successfully.")
    return pipe, fantasytalking, wav2vec_processor, wav2vec


# Add cancel_fn and new args
def main(
    args: dict,
    pipe: WanVideoPipeline,
    fantasytalking: FantasyTalkingAudioConditionModel,
    wav2vec_processor: Wav2Vec2Processor,
    wav2vec: Wav2Vec2Model,
    cancel_fn=None, # Added cancel_fn parameter
    gradio_progress=None # Added Gradio progress object parameter
    ):
    """Generates the video based on the provided arguments and loaded models."""
    start_time_global = datetime.now() # Record overall start time
    start_time_perf = time.perf_counter() # More precise timer

    # --- Get Generation Index/Total and Prompt Index for Logging ---
    # Use .get() with defaults for safety
    generation_index = args.get('generation_index', 1) # Variation index
    total_generations = args.get('total_generations', 1) # Total variations for this prompt
    prompt_index = args.get('prompt_index') # Prompt index (1-based or None) <<< Added

    log_prefix = f"[Gen {generation_index}/{total_generations}]"
    if prompt_index is not None:
        log_prefix += f"[Prompt {prompt_index}]" # <<< Add prompt index to log prefix

    print(f"\n{log_prefix} ===== Starting Generation Task =====")
    print(f"{log_prefix} Start time: {start_time_global.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Log Received Arguments ---
    print(f"{log_prefix} Received arguments:")
    # Use pprint for better readability, exclude potentially large objects
    printable_args = {}
    for k, v in args.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            printable_args[k] = f"<{type(v).__name__} shape={getattr(v, 'shape', 'N/A')} device={getattr(v, 'device', 'N/A')}>"
        elif isinstance(v, Image.Image):
            printable_args[k] = f"<PIL.Image size={getattr(v, 'size', 'N/A')} mode={getattr(v, 'mode', 'N/A')}>"
        elif isinstance(v, Path): # Convert Path objects to strings for printing
            printable_args[k] = str(v)
        else:
            printable_args[k] = v
    pprint.pprint(printable_args, indent=2, width=120) # Adjust width if needed


    # --- Parameter Extraction and Validation ---
    try:
        output_dir_str = args['output_dir']
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"{log_prefix} Ensured output directory exists: {output_dir}")
        USED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        print(f"{log_prefix} Ensured used audios directory exists: {USED_AUDIO_DIR}")

        # Filenaming strategy depends on 'output_base_name' (batch mode) or sequential counter
        output_base_name_arg = args.get('output_base_name') # For batch mode override stem
        save_metadata = args.get('save_metadata', False)
        # Indices already retrieved above for logging

        # Initialize path variables
        base_name = None
        base_name_with_suffixes = None # <<< Renamed for clarity
        save_path = None
        metadata_path = None
        save_path_tmp = None
        used_audio_filename = None
        saved_audio_path_str = None

        # Define suffixes based on indices
        prompt_suffix = f"_prompt{prompt_index}" if prompt_index is not None else ""
        variation_suffix = f"_{generation_index:04d}" if total_generations > 1 else ""

        if output_base_name_arg:
            # Batch mode: Base name is provided stem. Combine with suffixes.
            base_name = output_base_name_arg # The core stem from image filename
            base_name_with_suffixes = f"{base_name}{prompt_suffix}{variation_suffix}"
            print(f"{log_prefix} Batch mode: Determined base name with suffixes: {base_name_with_suffixes}")
        else:
            # Single/Sequential mode: Find the NEXT available 4-digit prefix.
            print(f"{log_prefix} Sequential mode: Scanning for highest existing sequence number in {output_dir}...")
            max_sequence_num = 0
            # Regex to find filenames starting with 4 digits, followed by optional suffixes, ending in .mp4 or .txt
            sequence_pattern = re.compile(r"^(\d{4}).*?\.(mp4|txt)$")

            try:
                # Ensure output directory exists before scanning
                output_dir.mkdir(parents=True, exist_ok=True)
                for filename in os.listdir(output_dir):
                    match = sequence_pattern.match(filename)
                    if match:
                        num = int(match.group(1))
                        if num > max_sequence_num:
                            max_sequence_num = num
                print(f"{log_prefix} Highest sequence number found: {max_sequence_num}")
            except FileNotFoundError:
                # This case should be less likely now due to mkdir above, but keep for safety
                print(f"{log_prefix} Output directory was not found during scan. Starting sequence from 1.")
                max_sequence_num = 0 # Ensure it starts at 1 if dir doesn't exist
            except Exception as e:
                print(f"{log_prefix} Warning: Error scanning output directory for sequence numbers: {e}. Starting sequence from 1.")
                traceback.print_exc() # Log the full error for debugging
                max_sequence_num = 0 # Default to 0 on other errors

            # Determine the base name for THIS generation task (all variations share this base)
            next_sequence_num = max_sequence_num + 1
            base_name = f"{next_sequence_num:04d}"
            print(f"{log_prefix} Sequential mode: Using base sequence number '{base_name}' for this generation task.")

            # Construct the final name stem for THIS specific file (base + suffixes)
            base_name_with_suffixes = f"{base_name}{prompt_suffix}{variation_suffix}"
            print(f"{log_prefix} Sequential mode: Final name stem for this specific file: '{base_name_with_suffixes}'")


        # Extract other parameters
        num_frames = args['num_frames']
        fps = args['fps'] # This is the original FPS
        audio_path = args['audio_path']
        image_path = args['image_path']
        width = args['width']
        height = args['height']
        seed = args['seed']
        save_video_quality = args.get('save_video_quality', 18) # Use CRF, default 18

        # <<< Extract RIFE parameters >>>
        rife_mode = args.get('rife_mode', "None")
        rife_max_fps = args.get('rife_max_fps', 30)
        # Ensure original_fps is available if needed for RIFE calculations
        original_fps = args.get('original_fps', fps) # Fallback to fps if not passed

        # <<< Use actual prompt passed for this specific run >>>
        current_prompt = args['prompt'] # This is the single line prompt for this run

        print(f"{log_prefix} Source Image: {Path(image_path).name}")
        print(f"{log_prefix} Source Audio: {Path(audio_path).name}")
        print(f"{log_prefix} Current Prompt: '{current_prompt[:100]}{'...' if len(current_prompt)>100 else ''}'") # Log truncated prompt
        print(f"{log_prefix} Target dims: {width}x{height}, Frames: {num_frames}, FPS: {fps}, Seed: {seed}, Quality (CRF): {save_video_quality}")

    except KeyError as e: print(f"{log_prefix} !!! Error: Missing required argument: {e}"); raise ValueError(f"Missing required argument: {e}") from e
    except Exception as e: print(f"{log_prefix} !!! Error during parameter extraction: {e}"); traceback.print_exc(); raise


    # --- Load and Resize Image ---
    try:
        print(f"{log_prefix} Loading/resizing image: {image_path} to {width}x{height}")
        image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        print(f"{log_prefix} Resized image size: {image.size}")
    except FileNotFoundError: print(f"{log_prefix} !!! Error: Input image not found: {image_path}"); raise
    except Exception as e: print(f"{log_prefix} !!! Error loading/resizing image: {e}"); traceback.print_exc(); raise


    # --- Extract Audio Features ---
    device = pipe.device if pipe else "cuda"; print(f"{log_prefix} Target device for audio: {device}")
    try:
        print(f"{log_prefix} Extracting audio features (FPS={fps}, NumFrames={num_frames})...")
        if not Path(audio_path).exists(): raise FileNotFoundError(f"Audio file not found: {audio_path}")
        actual_audio_duration = librosa.get_duration(filename=audio_path); print(f"{log_prefix} Confirmed audio duration: {actual_audio_duration:.2f}s")
        if actual_audio_duration <= 0: raise ValueError(f"Audio file has zero/negative duration: {audio_path}")

        audio_wav2vec_fea = get_audio_features(wav2vec, wav2vec_processor, audio_path, fps, num_frames)
        print(f"{log_prefix} Audio Wav2Vec features extracted, shape: {audio_wav2vec_fea.shape}, device: {audio_wav2vec_fea.device}")
    except FileNotFoundError as e: print(f"{log_prefix} !!! Error: Audio file not found: {e}"); raise
    except Exception as e: print(f"{log_prefix} !!! Error extracting audio features: {e}"); traceback.print_exc(); raise


    # --- Process Audio Features (Projection & Splitting) ---
    try:
        print(f"{log_prefix} Projecting audio features...")
        audio_proj_fea = fantasytalking.get_proj_fea(audio_wav2vec_fea)
        print(f"{log_prefix} Audio projected features calculated, shape: {audio_proj_fea.shape}")
        print(f"{log_prefix} Calculating audio split points for num_frames={num_frames}...")
        pos_idx_ranges = fantasytalking.split_audio_sequence(audio_proj_fea.size(1), num_frames=num_frames)
        print(f"{log_prefix} Splitting audio features tensor...")
        audio_proj_split, audio_context_lens = fantasytalking.split_tensor_with_padding(audio_proj_fea, pos_idx_ranges, expand_length=4)
        print(f"{log_prefix} Audio features split, shape: {audio_proj_split.shape}, context lengths: {audio_context_lens}")
    except Exception as e: print(f"{log_prefix} !!! Error processing projected audio features: {e}"); traceback.print_exc(); raise


    # --- Prepare Diffusion Pipeline Arguments ---
    print(f"{log_prefix} Preparing arguments for WanVideoPipeline...")
    try:
        latents_num_frames = (num_frames - 1) // 4 + 1; print(f"{log_prefix} Calculated latents_num_frames: {latents_num_frames}")
        pipe_kwargs = {
            "prompt": current_prompt, # <<< Use the specific prompt for this run
            "negative_prompt": args['negative_prompt'],
            "input_image": image,
            "width": width, "height": height, "num_frames": num_frames,
            "num_inference_steps": args['inference_steps'], "seed": seed,
            "tiled": args['tiled_vae'],
            "audio_scale": args['audio_weight'],
            "cfg_scale": args['prompt_cfg_scale'],
            "audio_cfg_scale": args['audio_cfg_scale'],
            "audio_proj": audio_proj_split.to(device=pipe.device, dtype=torch.bfloat16), # <<< Use pipe's dtype
            "audio_context_lens": audio_context_lens,
            "latents_num_frames": latents_num_frames,
            "denoising_strength": args.get('denoising_strength', 1.0),
            "sigma_shift": args.get('sigma_shift', 5.0),
        }
        if args['tiled_vae']:
            try:
                pipe_kwargs["tile_size"] = (int(args['tile_size_h']), int(args['tile_size_w']))
                pipe_kwargs["tile_stride"] = (int(args['tile_stride_h']), int(args['tile_stride_w']))
                print(f"{log_prefix} Tiling enabled: size={pipe_kwargs['tile_size']}, stride={pipe_kwargs['tile_stride']}")
            except (ValueError, TypeError, KeyError) as e: print(f"{log_prefix} Warning: Invalid tiling params ({e}). Disabling tiling."); pipe_kwargs["tiled"] = False
        else: print(f"{log_prefix} Tiling disabled.")

        print(f"{log_prefix} Pipeline arguments prepared (excluding large tensors/images):")
        printable_pipe_kwargs = {k: f"<Tensor shape={v.shape} dtype={v.dtype} dev={v.device}>" if isinstance(v, torch.Tensor) else (f"<PIL Image {v.size} {v.mode}>" if isinstance(v, Image.Image) else v) for k, v in pipe_kwargs.items()}
        pprint.pprint(printable_pipe_kwargs, indent=2, width=120)

    except KeyError as e: print(f"{log_prefix} !!! Error: Missing argument for pipeline: {e}"); raise ValueError(f"Missing argument for pipeline: {e}") from e
    except Exception as e: print(f"{log_prefix} !!! Error preparing pipeline arguments: {e}"); traceback.print_exc(); raise


    # --- Image-to-Video Diffusion Pipeline Execution ---
    print(f"{log_prefix} Starting WanVideoPipeline generation ({pipe_kwargs['num_inference_steps']} steps)...")
    video_frames_pil = None; pipeline_start_time = time.perf_counter()
    try:
        video_frames_pil = pipe(**pipe_kwargs, cancel_fn=cancel_fn, progress_bar_cmd=tqdm, gradio_progress=gradio_progress)
        pipeline_duration = time.perf_counter() - pipeline_start_time
        print(f"{log_prefix} WanVideoPipeline finished successfully in {pipeline_duration:.2f}s.")
        if isinstance(video_frames_pil, list) and len(video_frames_pil) > 0 and isinstance(video_frames_pil[0], Image.Image): print(f"{log_prefix} Received {len(video_frames_pil)} PIL frames.")
        else: print(f"{log_prefix} Warning: Pipeline returned unexpected type/empty list: {type(video_frames_pil)}"); raise ValueError("Pipeline did not return list of PIL Images.")

        # --- Replace First Frame with Input Image ---
        if isinstance(video_frames_pil, list) and len(video_frames_pil) > 0: print(f"{log_prefix} Replacing first frame with input image."); video_frames_pil[0] = image
        else: print(f"{log_prefix} Warning: Cannot replace first frame (video_frames_pil invalid).")

    except CancelledError as e:
        pipeline_duration = time.perf_counter() - pipeline_start_time; print(f"{log_prefix} Pipeline cancelled after {pipeline_duration:.2f}s. Message: {e}"); raise
    except Exception as e:
        pipeline_duration = time.perf_counter() - pipeline_start_time; print(f"{log_prefix} !!! Critical Error during Pipeline execution after {pipeline_duration:.2f}s: {e}"); traceback.print_exc()
        if torch.cuda.is_available(): print(f"{log_prefix} Clearing CUDA cache after error..."); torch.cuda.empty_cache(); print(f"{log_prefix} Cache clear attempted.")
        raise


    # --- Convert PIL Frames to Numpy Array ---
    try:
        print(f"{log_prefix} Converting {len(video_frames_pil)} PIL frames to NumPy array...")
        video_array = np.stack([np.array(frame.convert("RGB")) for frame in video_frames_pil])
        print(f"{log_prefix} Video array shape: {video_array.shape}, dtype: {video_array.dtype}")
        if video_array.shape[1] != height or video_array.shape[2] != width: print(f"{log_prefix} Warning: Frame dims ({video_array.shape[1:3]}) != requested ({height}x{width}).")
    except Exception as e: print(f"{log_prefix} !!! Error converting PIL frames to NumPy: {e}"); traceback.print_exc(); raise


    # --- Determine Filename for Single/Sequential Mode & Define All Paths ---
    # <<< REMOVED redundant sequential naming logic block >>>
    # <<< The logic determining base_name and base_name_with_suffixes now happens earlier (lines ~253-292) >>>
    # <<< Paths are now defined using the base_name_with_suffixes calculated there. >>>

    # Ensure base_name_with_suffixes was determined earlier
    if base_name_with_suffixes is None:
        # This should not happen if secourses_app.py provides output_base_name
        # or if the initial sequential logic ran correctly.
        raise RuntimeError(f"{log_prefix} Failed to determine a valid base_name_with_suffixes earlier in the script.")

    print(f"{log_prefix} Using determined base name with suffixes: '{base_name_with_suffixes}'")
    print(f"{log_prefix} Determining final output paths based on this name...")
    audio_path_obj = Path(audio_path)

    # Define paths using the final base_name_with_suffixes (determined earlier)
    save_path = output_dir / f"{base_name_with_suffixes}.mp4"
    metadata_path = None # Initialize
    if save_metadata:
        metadata_path = output_dir / f"{base_name_with_suffixes}.txt"

    # Temporary file name still needs to be unique per process/time
    tmp_suffix = f"tmp_{base_name_with_suffixes}_{int(time.time())}_{os.getpid()}.mp4"
    save_path_tmp = output_dir / tmp_suffix

    # Use base_name_with_suffixes for the copied audio name as well
    used_audio_filename = f"{base_name_with_suffixes}{audio_path_obj.suffix}"
    used_audio_dest_path = USED_AUDIO_DIR / used_audio_filename

    print(f"{log_prefix} Final Save Path: {save_path}")
    if metadata_path: print(f"{log_prefix} Metadata Save Path: {metadata_path}")
    print(f"{log_prefix} Temporary Video Path: {save_path_tmp}")
    print(f"{log_prefix} Copied Input Audio Path: {used_audio_dest_path}")
    # --- End Filename Determination ---


    # --- Save Input Audio Copy --- #
    saved_audio_path_str = None
    try: shutil.copy2(audio_path, used_audio_dest_path); saved_audio_path_str = str(used_audio_dest_path.resolve()); print(f"{log_prefix} Copied input audio to: {used_audio_dest_path}")
    except Exception as copy_e: print(f"{log_prefix} !!! Warning: Failed to copy input audio {audio_path} to {used_audio_dest_path}: {copy_e}"); traceback.print_exc()


    # --- Save Temporary Video ---
    try: print(f"{log_prefix} Saving frames to temporary video: {save_path_tmp}"); save_video(video_array, str(save_path_tmp), fps=fps); print(f"{log_prefix} Temporary video saved.")
    except Exception as e: print(f"{log_prefix} !!! Error saving temporary video: {e}"); traceback.print_exc(); raise


    # --- Combine Video and Audio using FFmpeg ---
    print(f"{log_prefix} Merging video and audio using FFmpeg (CRF: {save_video_quality})...")
    final_command = ["ffmpeg", "-y", "-i", str(save_path_tmp), "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "slow", "-crf", str(save_video_quality), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(save_path)]
    print(f"{log_prefix} Running FFmpeg command: {' '.join(final_command)}")
    ffmpeg_start_time = time.perf_counter()
    try:
        process = subprocess.run(final_command, check=True, capture_output=True, text=True, encoding='utf-8')
        ffmpeg_duration = time.perf_counter() - ffmpeg_start_time; print(f"{log_prefix} FFmpeg completed successfully in {ffmpeg_duration:.2f}s.")
        # print(f"{log_prefix} FFmpeg stderr:\n{process.stderr}") # Optional: log stderr on success too
    except subprocess.CalledProcessError as e:
        ffmpeg_duration = time.perf_counter() - ffmpeg_start_time; print(f"{log_prefix} !!! Error during FFmpeg (Exit code: {e.returncode}) after {ffmpeg_duration:.2f}s"); print(f"{log_prefix} Failed command: {' '.join(e.cmd)}"); print(f"{log_prefix} FFmpeg stderr:\n{e.stderr}")
        print(f"{log_prefix} FFmpeg failed, temp file kept: {save_path_tmp}")
        error_summary = e.stderr.splitlines()[-5:]; raise Exception(f"FFmpeg failed. Temp: {save_path_tmp}. Error: {' '.join(error_summary)}") from e
    except FileNotFoundError: print(f"{log_prefix} !!! Error: 'ffmpeg' not found."); raise Exception("'ffmpeg' not found. Install ffmpeg.")
    except Exception as e: print(f"{log_prefix} !!! Unexpected error during FFmpeg: {e}"); traceback.print_exc(); raise


    # --- Clean up temporary file ---
    if save_path.exists():
        try: print(f"{log_prefix} Removing temporary file: {save_path_tmp}"); os.remove(save_path_tmp); print(f"{log_prefix} Temp file removed.")
        except OSError as e: print(f"{log_prefix} Warning: Could not remove temp file {save_path_tmp}: {e}")
    else: print(f"{log_prefix} Warning: Final output {save_path} not found after FFmpeg. Temp file {save_path_tmp} may be kept.")

    # --- <<< RIFE Post-Processing (if enabled and FFmpeg successful) >>> ---
    rife_applied = False
    rife_output_path_final = None # Store the final RIFE path if successful
    if save_path.exists() and rife_mode != "None":
        print(f"{log_prefix} [RIFE] Starting RIFE post-processing ({rife_mode}) for: {save_path}")
        rife_start_time = time.perf_counter()

        base_video_path = Path(save_path) # Use the path from FFmpeg
        rife_multiplier = 0
        rife_suffix = ""
        if rife_mode == "2x FPS":
            rife_multiplier = 2
            rife_suffix = "_RIFE_2x"
        elif rife_mode == "4x FPS":
            rife_multiplier = 4
            rife_suffix = "_RIFE_4x"

        if rife_multiplier > 0:
            rife_output_name = f"{base_video_path.stem}{rife_suffix}{base_video_path.suffix}"
            rife_output_path = base_video_path.parent / rife_output_name

            # Calculate target FPS and check against limit
            try:
                # Use the original_fps extracted earlier
                target_rife_fps = original_fps * rife_multiplier
                print(f"{log_prefix} [RIFE] Original FPS: {original_fps}, Target RIFE FPS: {target_rife_fps}, Limit: {rife_max_fps}")

                rife_fps_arg = None
                if rife_max_fps and target_rife_fps > rife_max_fps:
                    rife_fps_arg = int(rife_max_fps) # Limit the FPS
                    print(f"{log_prefix} [RIFE] Target FPS ({target_rife_fps}) exceeds limit ({rife_max_fps}). Setting RIFE FPS to {rife_fps_arg}.")
                else:
                    print(f"{log_prefix} [RIFE] Target FPS ({target_rife_fps}) within limit. Using multiplier {rife_multiplier}x.")
                    # No --fps needed, RIFE will use --multi

                # Build the command
                rife_script_path_relative = "Practical-RIFE/inference_video.py" # Relative path
                # Ensure the script exists relative to the workspace root
                rife_script_path_abs = Path(rife_script_path_relative).resolve()
                rife_cwd = rife_script_path_abs.parent # CWD should be the directory containing the script

                if not rife_script_path_abs.exists():
                    raise FileNotFoundError(f"RIFE script not found at {rife_script_path_abs}")

                cmd = [
                    sys.executable, # Use the current Python interpreter
                    str(rife_script_path_abs.name), # Just the script name
                    "--video", str(base_video_path.resolve()),
                    "--output", str(rife_output_path.resolve()),
                    "--multi", str(rife_multiplier),
                    "--ext", "mp4" # Force mp4 output
                ]
                if rife_fps_arg is not None:
                    cmd.extend(["--fps", str(rife_fps_arg)])

                print(f"{log_prefix} [RIFE] Running command in {rife_cwd}: {' '.join(cmd)}")
                try:
                    # Run RIFE process
                    process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', cwd=str(rife_cwd))
                    rife_duration = time.perf_counter() - rife_start_time
                    print(f"{log_prefix} [RIFE] stdout:\n{process.stdout}") # Log RIFE stdout
                    print(f"{log_prefix} [RIFE] stderr:\n{process.stderr}") # Log RIFE stderr
                    print(f"{log_prefix} [RIFE] Processing finished successfully in {rife_duration:.2f}s.")

                    # <<< IMPORTANT: Update save_path if RIFE succeeded >>>
                    save_path = rife_output_path # Update to the RIFE output path
                    rife_applied = True
                    rife_output_path_final = rife_output_path # Store for metadata
                    print(f"{log_prefix} [RIFE] Updated save_path to: {save_path}")

                    # Optional: Delete the original non-RIFE file?
                    # try:
                    #     print(f"{log_prefix} [RIFE] Deleting original non-RIFE file: {base_video_path}")
                    #     base_video_path.unlink()
                    # except OSError as del_e:
                    #     print(f"{log_prefix} [RIFE] Warning: Could not delete original file {base_video_path}: {del_e}")

                except FileNotFoundError as fnf_e:
                    # This could happen if sys.executable is wrong, or the script disappears between check and run
                    print(f"{log_prefix} !!! Error [RIFE] Command failed (FileNotFoundError): {fnf_e}. Is Python path correct? Is script accessible? CWD: {rife_cwd}")
                    # Keep original path, rife_applied = False
                except subprocess.CalledProcessError as e:
                    rife_duration = time.perf_counter() - rife_start_time
                    print(f"{log_prefix} !!! Error [RIFE] Processing failed after {rife_duration:.2f}s. Exit code: {e.returncode}")
                    print(f"{log_prefix} [RIFE] Command: {' '.join(e.cmd)}")
                    print(f"{log_prefix} [RIFE] stdout:\n{e.stdout}")
                    print(f"{log_prefix} [RIFE] stderr:\n{e.stderr}")
                    # Keep original path, rife_applied = False
                except Exception as e:
                    rife_duration = time.perf_counter() - rife_start_time
                    print(f"{log_prefix} !!! Error [RIFE] An unexpected error occurred during RIFE processing after {rife_duration:.2f}s: {e}")
                    traceback.print_exc()
                    # Keep original path, rife_applied = False

            except FileNotFoundError as e:
                # Error finding the script *before* running subprocess
                 print(f"{log_prefix} !!! Error [RIFE] Setup failed: {e}. Skipping RIFE.")
            except Exception as e:
                print(f"{log_prefix} !!! Error [RIFE] Error preparing RIFE command or calculating FPS for {base_video_path.name}: {e}")
                traceback.print_exc()
                # Keep original path, rife_applied = False
    elif not save_path.exists():
        print(f"{log_prefix} Skipping RIFE because FFmpeg output {save_path.name} was not found.")
    else: # rife_mode is None
        print(f"{log_prefix} RIFE is disabled (mode={rife_mode}). Skipping RIFE post-processing.")
    # --- End RIFE Post-Processing --- #

    # --- Metadata Saving Logic ---
    if save_metadata and metadata_path:
        print(f"{log_prefix} Saving metadata to {metadata_path}...")
        try:
            end_time_global = datetime.now(); end_time_perf = time.perf_counter(); generation_duration_perf = end_time_perf - start_time_perf; generation_duration_wall = end_time_global - start_time_global
            metadata = {
                "generation_timestamp_utc": start_time_global.utcnow().isoformat() + "Z",
                "generation_duration_seconds": round(generation_duration_perf, 3),
                "generation_wall_time_seconds": round(generation_duration_wall.total_seconds(), 3),
                "output_video_file": save_path.name, # <<< Will reflect RIFE output if applied
                "output_metadata_file": metadata_path.name if metadata_path else "N/A",
                "base_name_with_suffixes": base_name_with_suffixes, # <<< Use the name including suffixes
                "input_image_file": Path(args['image_path']).name,
                "input_audio_file": Path(args['audio_path']).name,
                "saved_input_audio_copy": saved_audio_path_str,
                "variation_index": generation_index, # <<< Renamed for clarity
                "total_variations_for_prompt": total_generations, # <<< Renamed for clarity
                "prompt_index": prompt_index, # <<< Added prompt index (1-based or None)
                "rife_applied": rife_applied, # <<< Added RIFE status
                "rife_mode": rife_mode, # <<< Added RIFE mode used
                "rife_output_file": rife_output_path_final.name if rife_output_path_final else "N/A", # <<< Added RIFE output name
                "settings": {
                     "prompt": current_prompt, # <<< Save the specific prompt used
                     "negative_prompt": args['negative_prompt'],
                     "width": args['width'], "height": args['height'], "num_frames": args['num_frames'], "fps": args['fps'],
                     "audio_weight": args['audio_weight'], "prompt_cfg_scale": args['prompt_cfg_scale'], "audio_cfg_scale": args['audio_cfg_scale'],
                     "inference_steps": args['inference_steps'], "seed": args['seed'],
                     "tiled_vae": args['tiled_vae'],
                     "tile_size_h": args.get('tile_size_h', 'N/A'), "tile_size_w": args.get('tile_size_w', 'N/A'),
                     "tile_stride_h": args.get('tile_stride_h', 'N/A'), "tile_stride_w": args.get('tile_stride_w', 'N/A'),
                     "sigma_shift": args.get('sigma_shift', 'N/A'), "denoising_strength": args.get('denoising_strength', 'N/A'),
                     "save_video_quality_crf": save_video_quality,
                     # Add pipeline dtype? Needs passing from app
                     # "torch_dtype_pipeline": str(pipe.torch_dtype) if pipe else 'N/A',
                }
            }
            with open(metadata_path, 'w', encoding='utf-8') as f: json.dump(metadata, f, indent=4, ensure_ascii=False)
            print(f"{log_prefix} Metadata saved successfully.")
        except Exception as meta_e: print(f"{log_prefix} !!! Warning: Failed to save metadata to {metadata_path}: {meta_e}"); traceback.print_exc()
    elif save_metadata and not metadata_path: print(f"{log_prefix} Info: Metadata saving enabled but path not determined. Skipping.")
    else: print(f"{log_prefix} Metadata saving disabled. Skipping.")
    # --- End Metadata Saving ---


    total_duration = time.perf_counter() - start_time_perf
    final_output_path_to_return = save_path # Default to the path after ffmpeg
    if rife_applied and rife_output_path_final and rife_output_path_final.exists():
        final_output_path_to_return = rife_output_path_final # Update to RIFE path if successful
        print(f"{log_prefix} Returning RIFE output path: {final_output_path_to_return}")
    else:
        print(f"{log_prefix} Returning original/FFmpeg output path: {final_output_path_to_return}")

    print(f"{log_prefix} ===== Finished Generation Task (Total duration: {total_duration:.2f}s) =====")

    return str(final_output_path_to_return) # Return final path as string

# --- END OF FILE infer.py ---