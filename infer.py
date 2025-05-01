# --- START OF FILE infer.py ---

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
        wan_loading_dtype = torch_dtype
        print(f"[Model Loading] Selected Wan I2V model for BF16/FP16 pipeline: {wan_model_filename} (loading as {torch_dtype})")
    elif target_pipeline_dtype == torch.float8_e4m3fn:
        wan_model_filename = "wan21_i2v_720p_14B_fp8_e4m3fn.safetensors"
        wan_loading_dtype = torch_dtype # Load FP8 model as FP8
        print(f"[Model Loading] Selected Wan I2V model for FP8 pipeline: {wan_model_filename} (loading as {torch_dtype})")
    else:
        # Fallback or error - defaulting to FP8 might be risky, but let's match previous behavior slightly safer
        print(f"[Warning][Model Loading] Unsupported torch_dtype ({target_pipeline_dtype}) received. Defaulting to FP8 model. Behavior might be unexpected.")
        wan_model_filename = "wan21_i2v_720p_14B_fp8_e4m3fn.safetensors"
        wan_loading_dtype = torch.float16

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
        if target_pipeline_dtype is torch.bfloat16:
            pipe = WanVideoPipeline.from_model_manager(
                model_manager, torch_dtype=torch.bfloat16, device=device  # Pipeline manages device movement
            )
        else:
            pipe = WanVideoPipeline.from_model_manager(
                model_manager, torch_dtype=torch.float16, device=device  # Pipeline manages device movement
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

    # --- Get Generation Index/Total for Logging ---
    # Use .get() with defaults for safety
    generation_index = args.get('generation_index', 1)
    total_generations = args.get('total_generations', 1)
    log_prefix = f"[Gen {generation_index}/{total_generations}]"
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
        # --- Ensure Used Audios Directory Exists --- #
        USED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        print(f"{log_prefix} Ensured used audios directory exists: {USED_AUDIO_DIR}")
        # ----------------------------------------- #

        # Filenaming strategy depends on 'output_base_name'
        output_base_name_arg = args.get('output_base_name') # For batch mode override
        save_metadata = args.get('save_metadata', False) # Check if metadata saving is enabled

        base_name = ""
        save_path = None
        metadata_path = None

        if output_base_name_arg:
            # Batch mode: Use the provided stem directly
            base_name = output_base_name_arg
            print(f"{log_prefix} Using provided base name for batch mode: {base_name}")
            save_path = output_dir / f"{base_name}.mp4"
            if save_metadata:
                 metadata_path = output_dir / f"{base_name}.txt"
            # Skip check is handled in the calling function (secourses_app.py)
            if save_path.exists():
                 print(f"{log_prefix} Info: Output file {save_path} already exists (batch mode). FFmpeg will overwrite.")
        else:
            # Single/Sequential mode: Find the next available 000X number
            print(f"{log_prefix} Determining sequential output filename (e.g., 0001.mp4)...")
            sequence_num = 1
            while True:
                base_name = f"{sequence_num:04d}"
                potential_save_path = output_dir / f"{base_name}.mp4"
                potential_metadata_path = output_dir / f"{base_name}.txt" if save_metadata else None

                video_exists = potential_save_path.exists()
                metadata_exists = potential_metadata_path is not None and potential_metadata_path.exists()

                if not video_exists and not metadata_exists:
                    save_path = potential_save_path
                    metadata_path = potential_metadata_path # Will be None if save_metadata is False
                    print(f"{log_prefix} Using unique sequential base name: {base_name}")
                    break

                # Log why we are skipping this number
                reason = []
                if video_exists: reason.append(f"video '{base_name}.mp4'")
                if metadata_exists: reason.append(f"metadata '{base_name}.txt'")
                #print(f"{log_prefix} Skipping sequence {sequence_num}: {' and '.join(reason)} already exist(s).")

                sequence_num += 1
                if sequence_num > 99999: # Increased safety break limit
                    raise RuntimeError(f"{log_prefix} Could not find a unique filename after {sequence_num-1} attempts in {output_dir}. Please clean the directory or increase the limit.")

        if not save_path: # Safety check
             raise RuntimeError(f"{log_prefix} Failed to determine a valid output save path.")

        # Define temporary path using the determined base_name
        # Add timestamp and process ID for better uniqueness during parallel runs (though UI prevents this)
        tmp_suffix = f"tmp_{base_name}_{int(time.time())}_{os.getpid()}.mp4"
        save_path_tmp = output_dir / tmp_suffix

        print(f"{log_prefix} Final Save Path: {save_path}")
        if metadata_path:
            print(f"{log_prefix} Metadata Save Path: {metadata_path}")
        print(f"{log_prefix} Temporary Video Path: {save_path_tmp}")

        # Extract other parameters
        num_frames = args['num_frames']
        fps = args['fps']
        audio_path = args['audio_path']
        image_path = args['image_path']
        width = args['width']
        height = args['height']
        seed = args['seed']
        save_video_quality = args.get('save_video_quality', 18) # Use CRF, default 18

        print(f"{log_prefix} Source Image: {Path(image_path).name}")
        print(f"{log_prefix} Source Audio: {Path(audio_path).name}")
        print(f"{log_prefix} Target dims: {width}x{height}, Frames: {num_frames}, FPS: {fps}, Seed: {seed}, Quality (CRF): {save_video_quality}")

    except KeyError as e:
        print(f"{log_prefix} !!! Error: Missing required argument: {e}")
        raise ValueError(f"Missing required argument in args dictionary: {e}") from e
    except Exception as e:
        print(f"{log_prefix} !!! Error during parameter extraction or path setup: {e}")
        traceback.print_exc()
        raise # Re-raise critical setup error


    # --- Load and Resize Image ---
    try:
        print(f"{log_prefix} Loading and resizing image from: {image_path} to {width}x{height}")
        image = Image.open(image_path).convert("RGB")
        # Use LANCZOS for high-quality resize
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        print(f"{log_prefix} Resized image size: {image.size}")
    except FileNotFoundError:
         print(f"{log_prefix} !!! Error: Input image file not found at {image_path}")
         raise
    except Exception as e:
        print(f"{log_prefix} !!! Error loading or resizing image: {e}")
        traceback.print_exc()
        raise


    # --- Extract Audio Features ---
    # Device needs to be determined (assuming it's passed or inferred)
    device = pipe.device if pipe else "cuda" # Get device from pipeline if possible
    print(f"{log_prefix} Determined target device for audio processing: {device}")
    try:
        print(f"{log_prefix} Extracting audio features (Target FPS={fps}, Target NumFrames={num_frames})...")
        # Ensure audio file exists before processing
        if not Path(audio_path).exists():
             raise FileNotFoundError(f"Audio file not found: {audio_path}")
        # Check duration again right before processing, in case file changed
        actual_audio_duration = librosa.get_duration(filename=audio_path)
        if actual_audio_duration <= 0:
             raise ValueError(f"Audio file has zero or negative duration: {audio_path}")
        print(f"{log_prefix} Confirmed audio duration before feature extraction: {actual_audio_duration:.2f}s")

        audio_wav2vec_fea = get_audio_features(
            wav2vec, wav2vec_processor, audio_path, fps, num_frames 
        )
        print(f"{log_prefix} Audio Wav2Vec features extracted, shape: {audio_wav2vec_fea.shape}, device: {audio_wav2vec_fea.device}")
    except FileNotFoundError as e:
        print(f"{log_prefix} !!! Error: Audio file not found during feature extraction: {e}")
        raise
    except Exception as e:
        print(f"{log_prefix} !!! Error extracting audio features: {e}")
        traceback.print_exc()
        raise


    # --- Process Audio Features (Projection & Splitting) ---
    try:
        print(f"{log_prefix} Projecting audio features...")
        audio_proj_fea = fantasytalking.get_proj_fea(audio_wav2vec_fea)
        print(f"{log_prefix} Audio projected features calculated, shape: {audio_proj_fea.shape}")

        # Calculate split points based on the *actual number of frames* being generated
        print(f"{log_prefix} Calculating audio split points for num_frames={num_frames}...")
        pos_idx_ranges = fantasytalking.split_audio_sequence(
            audio_proj_fea.size(1), num_frames=num_frames # Use the final num_frames
        )
        print(f"{log_prefix} Splitting audio features tensor...")
        audio_proj_split, audio_context_lens = fantasytalking.split_tensor_with_padding(
            audio_proj_fea, pos_idx_ranges, expand_length=4 # Assuming expand_length=4 is standard
        )
        print(f"{log_prefix} Audio features split, shape: {audio_proj_split.shape}, context lengths calculated: {audio_context_lens}")
    except Exception as e:
        print(f"{log_prefix} !!! Error processing projected audio features: {e}")
        traceback.print_exc()
        raise


    # --- Prepare Diffusion Pipeline Arguments ---
    print(f"{log_prefix} Preparing arguments for WanVideoPipeline...")
    try:
        # Determine correct latent frame count based on num_frames
        latents_num_frames = (num_frames - 1) // 4 + 1
        print(f"{log_prefix} Calculated latents_num_frames: {latents_num_frames}")

        pipe_kwargs = {
            "prompt": args['prompt'],
            "negative_prompt": args['negative_prompt'],
            "input_image": image, # Use the resized PIL Image object
            "width": width,
            "height": height,
            "num_frames": num_frames, # Use the final calculated num_frames
            "num_inference_steps": args['inference_steps'],
            "seed": seed,
            "tiled": args['tiled_vae'], # Use the boolean value directly
            "audio_scale": args['audio_weight'], # Map from Gradio 'audio_weight'
            "cfg_scale": args['prompt_cfg_scale'],
            "audio_cfg_scale": args['audio_cfg_scale'],
            # Ensure audio features are on the correct device and dtype
            "audio_proj": audio_proj_split.to(device=pipe.device, dtype=torch.float16),
            "audio_context_lens": audio_context_lens,
            "latents_num_frames": latents_num_frames, # Pass calculated latent frames
            "denoising_strength": args.get('denoising_strength', 1.0), # Use .get for safety
            "sigma_shift": args.get('sigma_shift', 5.0), # Pass sigma_shift from args
            # cancel_fn will be passed separately below
        }

        if args['tiled_vae']:
            # Ensure tile sizes/strides are integers
            try:
                tile_size_h = int(args['tile_size_h'])
                tile_size_w = int(args['tile_size_w'])
                tile_stride_h = int(args['tile_stride_h'])
                tile_stride_w = int(args['tile_stride_w'])
                pipe_kwargs["tile_size"] = (tile_size_h, tile_size_w)
                pipe_kwargs["tile_stride"] = (tile_stride_h, tile_stride_w)
                print(f"{log_prefix} Tiling enabled with size: {pipe_kwargs['tile_size']}, stride: {pipe_kwargs['tile_stride']}")
            except (ValueError, TypeError, KeyError) as e:
                 print(f"{log_prefix} Warning: Invalid tiling parameters provided ({e}). Disabling tiling.")
                 pipe_kwargs["tiled"] = False # Disable if params invalid
        else:
            print(f"{log_prefix} Tiling disabled.")

        # Log prepared arguments (excluding large tensors/images)
        print(f"{log_prefix} Pipeline arguments prepared:")
        printable_pipe_kwargs = {}
        for k, v in pipe_kwargs.items():
            if isinstance(v, torch.Tensor):
                printable_pipe_kwargs[k] = f"<Tensor shape={v.shape} dtype={v.dtype} device={v.device}>"
            elif isinstance(v, Image.Image):
                 printable_pipe_kwargs[k] = f"<PIL.Image size={v.size} mode={v.mode}>"
            else:
                 printable_pipe_kwargs[k] = v
        pprint.pprint(printable_pipe_kwargs, indent=2, width=120)

    except KeyError as e:
        print(f"{log_prefix} !!! Error: Missing expected argument while preparing pipeline kwargs: {e}")
        raise ValueError(f"Missing argument for pipeline: {e}") from e
    except Exception as e:
        print(f"{log_prefix} !!! Error preparing pipeline arguments: {e}")
        traceback.print_exc()
        raise


    # --- Image-to-Video Diffusion Pipeline Execution ---
    print(f"{log_prefix} Starting WanVideoPipeline generation ({pipe_kwargs['num_inference_steps']} steps)...")
    video_frames_pil = None # Initialize variable
    pipeline_start_time = time.perf_counter()
    try:
        # *** Pass cancel_fn, tqdm, and gradio_progress to the pipeline call ***
        video_frames_pil = pipe(
            **pipe_kwargs,
            cancel_fn=cancel_fn,
            progress_bar_cmd=tqdm, # <<< Pass tqdm for console progress
            gradio_progress=gradio_progress # <<< Pass Gradio progress object
        )
        # *******************************************
        pipeline_duration = time.perf_counter() - pipeline_start_time
        print(f"{log_prefix} WanVideoPipeline finished successfully in {pipeline_duration:.2f}s.")
        if isinstance(video_frames_pil, list) and len(video_frames_pil) > 0 and isinstance(video_frames_pil[0], Image.Image):
             print(f"{log_prefix} Received {len(video_frames_pil)} PIL frames.")
        else:
             # This case should ideally not happen if pipe returns correctly
             print(f"{log_prefix} Warning: Pipeline returned unexpected type or empty list: {type(video_frames_pil)}")
             raise ValueError("Pipeline did not return a list of PIL Image frames.")

        # --- Replace First Frame with Input Image ---
        if isinstance(video_frames_pil, list) and len(video_frames_pil) > 0:
            print(f"{log_prefix} Replacing first generated frame with the input image.")
            video_frames_pil[0] = image # 'image' is the resized input PIL Image
        else:
             print(f"{log_prefix} Warning: Cannot replace first frame as video_frames_pil is not a non-empty list.")
        # ---------------------------------------------

    except CancelledError as e: # Catch the specific error from the pipeline
        pipeline_duration = time.perf_counter() - pipeline_start_time
        print(f"{log_prefix} Pipeline cancelled by user after {pipeline_duration:.2f}s. Message: {e}")
        # No model unloading here, handled by caller (secourses_app.py)
        raise # Re-raise CancelledError to be caught by Gradio app loop/handler
    except Exception as e:
        pipeline_duration = time.perf_counter() - pipeline_start_time
        print(f"{log_prefix} !!! Critical Error during WanVideoPipeline execution after {pipeline_duration:.2f}s: {e}")
        traceback.print_exc()
        # Attempt to clear cache even on general errors
        print(f"{log_prefix} Attempting to clear CUDA cache after pipeline error...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{log_prefix} CUDA cache clear attempted.")
        else:
            print(f"{log_prefix} CUDA not available, skipping cache clear.")
        raise # Re-raise the original exception


    # --- Convert PIL Frames to Numpy Array ---
    try:
        print(f"{log_prefix} Converting {len(video_frames_pil)} PIL frames to NumPy array...")
        # Ensure frames are RGB uint8 [0, 255]
        # Using np.stack which is generally efficient
        video_array = np.stack([np.array(frame.convert("RGB")) for frame in video_frames_pil])
        # Expected shape: (T, H, W, C)
        print(f"{log_prefix} Video array created with shape: {video_array.shape}, dtype: {video_array.dtype}")
        if video_array.shape[0] != num_frames:
             print(f"{log_prefix} Warning: Number of frames in array ({video_array.shape[0]}) doesn't match requested ({num_frames}).")
        if video_array.shape[1] != height or video_array.shape[2] != width:
             print(f"{log_prefix} Warning: Frame dimensions ({video_array.shape[1]}x{video_array.shape[2]}) don't match requested ({height}x{width}).")

    except Exception as e:
        print(f"{log_prefix} !!! Error converting PIL frames to NumPy array: {e}")
        traceback.print_exc()
        raise # Re-raise critical error

    # --- Save Input Audio Copy --- #
    # Do this *before* potentially failing FFmpeg step
    saved_audio_path_str = None # Track path for metadata
    try:
        audio_path_obj = Path(audio_path)
        used_audio_filename = f"{base_name}{audio_path_obj.suffix}"
        used_audio_dest_path = USED_AUDIO_DIR / used_audio_filename
        shutil.copy2(audio_path, used_audio_dest_path) # copy2 preserves metadata
        saved_audio_path_str = str(used_audio_dest_path.resolve())
        print(f"{log_prefix} Copied input audio to: {used_audio_dest_path}")
    except Exception as copy_e:
        print(f"{log_prefix} !!! Warning: Failed to copy input audio {audio_path} to {used_audio_dest_path}: {copy_e}")
        # Continue generation even if audio copy fails, but log it.
        traceback.print_exc() # Log full traceback for the warning
    # -------------------------- #


    # --- Save Temporary Video (using diffsynth's save_video or alternative) ---
    # Consider if save_video handles T,H,W,C format correctly and desired codec/quality
    try:
        print(f"{log_prefix} Saving generated frames to temporary video: {save_path_tmp}")
        # save_video might use OpenCV which expects BGR by default if writing directly
        # If save_video uses moviepy or ffmpeg backend, RGB might be fine. Check its implementation.
        # Assuming save_video handles RGB (T,H,W,C) numpy array:
        save_video(video_array, str(save_path_tmp), fps=fps) # Use default quality for temp
        # If save_video expects BGR, convert first:
        # save_video(video_array[..., ::-1], str(save_path_tmp), fps=fps)
        print(f"{log_prefix} Temporary video saved successfully.")
    except Exception as e:
        print(f"{log_prefix} !!! Error saving temporary video using save_video utility: {e}")
        traceback.print_exc()
        # As a fallback, could try saving with a different library here if needed
        raise # Re-raise critical error


    # --- Combine Video and Audio using FFmpeg ---
    print(f"{log_prefix} Merging video and audio using FFmpeg (CRF: {save_video_quality})...")
    # Use a robust ffmpeg command with error checking
    final_command = [
        "ffmpeg",
        "-y", # Overwrite output file without asking
        "-i", str(save_path_tmp), # Input temporary video
        "-i", audio_path, # Input original audio
        "-map", "0:v:0", # Map video stream from first input
        "-map", "1:a:0?", # Map audio stream from second input, '?' makes it optional
        "-c:v", "libx264", # Video codec (widely compatible)
        "-preset", "slow", # Encoding speed preset (good balance) - can be medium or slow for better compression
        "-crf", str(save_video_quality), # Constant Rate Factor (0=lossless, 18=high, 23=default, 51=low)
        "-pix_fmt", "yuv420p", # Pixel format for compatibility
        "-c:a", "aac",     # Audio codec (widely compatible)
        "-b:a", "192k",    # Audio bitrate (adjust if needed)
        "-shortest", # Finish encoding when the shortest input stream ends (usually audio)
        str(save_path), # Final output path (ensure it's a string)
    ]
    print(f"{log_prefix} Running FFmpeg command: {' '.join(final_command)}")
    ffmpeg_start_time = time.perf_counter()
    try:
        process = subprocess.run(
            final_command,
            check=True,        # Raise CalledProcessError on failure
            capture_output=True, # Capture stdout and stderr
            text=True,         # Decode stdout/stderr as text
            encoding='utf-8'   # Specify encoding for robustness
        )
        ffmpeg_duration = time.perf_counter() - ffmpeg_start_time
        print(f"{log_prefix} FFmpeg command executed successfully in {ffmpeg_duration:.2f}s.")
        # Optional: Log ffmpeg output for debugging success cases too
        # print(f"{log_prefix} FFmpeg stdout:\n{process.stdout}")
        # print(f"{log_prefix} FFmpeg stderr:\n{process.stderr}") # Often contains useful info
    except subprocess.CalledProcessError as e:
        ffmpeg_duration = time.perf_counter() - ffmpeg_start_time
        print(f"{log_prefix} !!! Error during FFmpeg processing after {ffmpeg_duration:.2f}s (Exit code: {e.returncode})")
        print(f"{log_prefix} Failed FFmpeg command: {' '.join(e.cmd)}")
        # Log stderr which usually contains the error message
        print(f"{log_prefix} FFmpeg stderr:\n{e.stderr}")
        # Keep the temp file if ffmpeg fails for debugging
        print(f"{log_prefix} FFmpeg failed, temporary file kept at: {save_path_tmp}")
        # Raise a more informative exception
        error_summary = e.stderr.splitlines()[-5:] # Get last few lines of stderr
        raise Exception(f"FFmpeg failed to merge video and audio. Check logs. Temp file: {save_path_tmp}. Error: {' '.join(error_summary)}") from e
    except FileNotFoundError:
        print(f"{log_prefix} !!! Error: 'ffmpeg' command not found. Please ensure ffmpeg is installed and in your system's PATH.")
        raise Exception("'ffmpeg' not found. Please install ffmpeg and ensure it's in the PATH.")
    except Exception as e:
        # Catch other potential errors during subprocess execution
        print(f"{log_prefix} !!! An unexpected error occurred during FFmpeg execution: {e}")
        traceback.print_exc()
        raise # Re-raise


    # --- Clean up temporary file only if FFmpeg succeeded and final file exists ---
    if save_path.exists():
        try:
            print(f"{log_prefix} Removing temporary file: {save_path_tmp}")
            os.remove(save_path_tmp)
            print(f"{log_prefix} Temporary file removed successfully.")
        except OSError as e:
            # This is not critical, just log a warning
            print(f"{log_prefix} Warning: Could not remove temporary file {save_path_tmp}: {e}")
    else:
         # This indicates a problem if FFmpeg didn't error but the file is missing
         print(f"{log_prefix} Warning: Final output file {save_path} not found after FFmpeg process. Temporary file {save_path_tmp} may be kept.")


    # --- Metadata Saving Logic ---
    if save_metadata and metadata_path: # Ensure path was set (i.e., saving enabled)
        print(f"{log_prefix} Saving metadata to {metadata_path}...")
        try:
            end_time_global = datetime.now()
            end_time_perf = time.perf_counter()
            generation_duration_perf = end_time_perf - start_time_perf
            generation_duration_wall = end_time_global - start_time_global

            # Create metadata dictionary - include essential args
            metadata = {
                "generation_timestamp_utc": start_time_global.utcnow().isoformat() + "Z",
                "generation_duration_seconds": round(generation_duration_perf, 3),
                "generation_wall_time_seconds": round(generation_duration_wall.total_seconds(), 3),
                "output_video_file": save_path.name,
                "output_metadata_file": metadata_path.name,
                "base_name": base_name, # The core name (e.g., 0001 or image_stem)
                "input_image_file": Path(args['image_path']).name,
                "input_audio_file": Path(args['audio_path']).name,
                "saved_input_audio_copy": saved_audio_path_str, # <<< Added path to copied audio
                "generation_index": generation_index,
                "total_generations": total_generations,
                # Include relevant settings from the input args dictionary for reproducibility
                "settings": {
                     "prompt": args['prompt'],
                     "negative_prompt": args['negative_prompt'],
                     "width": args['width'],
                     "height": args['height'],
                     "num_frames": args['num_frames'],
                     "fps": args['fps'],
                     "audio_weight": args['audio_weight'],
                     "prompt_cfg_scale": args['prompt_cfg_scale'],
                     "audio_cfg_scale": args['audio_cfg_scale'],
                     "inference_steps": args['inference_steps'],
                     "seed": args['seed'],
                     "tiled_vae": args['tiled_vae'],
                     "tile_size_h": args.get('tile_size_h', 'N/A' if not args['tiled_vae'] else None),
                     "tile_size_w": args.get('tile_size_w', 'N/A' if not args['tiled_vae'] else None),
                     "tile_stride_h": args.get('tile_stride_h', 'N/A' if not args['tiled_vae'] else None),
                     "tile_stride_w": args.get('tile_stride_w', 'N/A' if not args['tiled_vae'] else None),
                     "sigma_shift": args.get('sigma_shift', 'N/A'),
                     "denoising_strength": args.get('denoising_strength', 'N/A'),
                     "save_video_quality_crf": save_video_quality,
                     # Add info about VRAM/dtype if available in args, otherwise skip
                     # "torch_dtype": str(args.get('torch_dtype', 'N/A')), # Passed to load_models
                     # "num_persistent_params": args.get('num_persistent_param_in_dit', 'N/A'), # Passed to load_models
                }
            }

            # Write JSON metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            print(f"{log_prefix} Metadata saved successfully.")

        except Exception as meta_e:
            # Log error but don't fail the whole generation
            print(f"{log_prefix} !!! Warning: Failed to save metadata to {metadata_path}: {meta_e}")
            traceback.print_exc()
    elif save_metadata and not metadata_path:
         print(f"{log_prefix} Info: Metadata saving was enabled, but metadata path was not determined (this shouldn't happen). Skipping metadata save.")
    else:
         print(f"{log_prefix} Metadata saving disabled. Skipping.")
    # --- End Metadata Saving ---


    total_duration = time.perf_counter() - start_time_perf
    print(f"{log_prefix} ===== Finished Generation Task (Total duration: {total_duration:.2f}s) =====")

    # Return the path to the final generated video
    return str(save_path) # Return as string for Gradio

# --- END OF FILE infer.py ---