# This repo is only SECourses followers Download Link : https://www.patreon.com/posts/127855145

https://github.com/user-attachments/assets/3d58b64d-2f99-4ab3-9a81-200839a0869e

![screencapture-127-0-0-1-7860-2025-05-01-13_12_32](https://github.com/user-attachments/assets/69044597-29bc-49e2-b677-a544607c01c4)


# FantasyTalking SECourses App - Version Comparison & Improvements

This document outlines the significant improvements, new features, and changes made in the updated FantasyTalking application (`secourses_app.py`, `infer.py`, `wan_video.py`) compared to the previous version (`old_app.py`, `old_infer.py`, `old_wan_video.py`).

The new version represents a major overhaul focused on enhancing user experience, adding powerful features, improving performance control, and increasing robustness.

---

## Major New Features & Enhancements

*   **Batch Processing:** Process entire folders of images automatically, finding corresponding audio/prompts.
*   **Preset System:** Save and load all UI settings to easily switch between configurations. Remembers the last used preset.
*   **RIFE Frame Interpolation:** Optionally increase the output video's FPS (2x or 4x) using state-of-the-art AI interpolation after generation.
*   **Multi-Prompt Support:** Generate multiple videos sequentially from one image/audio by entering multiple prompts (one per line).
*   **Graceful Cancellation:** Added a "Cancel All" button to stop ongoing single or batch generations.
*   **Video-to-Audio Extraction:** Upload a video file to automatically extract its audio track for use as input.
*   **Metadata Saving:** Optionally save detailed generation parameters and timings to a `.txt` file alongside the video.
*   **Modernized UI:** Cleaner theme, better organization, and more informative tooltips.
*   **Performance Controls:** Select model precision (FP8/BF16) and configure VRAM usage (Tiled VAE, Persistent Parameters).

---

## Detailed Improvements

### User Interface & Experience (`secourses_app.py` vs. `old_app.py`)

*   **Modernized UI:** Adopted a cleaner `gr.themes.Soft()` theme and reorganized the layout using columns, rows, and accordions for better navigation and clarity (Inputs, Settings, Advanced, Performance, Batch, Presets, RIFE).
*   **Enhanced Input Options:**
    *   Added a dedicated video upload input that automatically extracts audio via FFmpeg.
    *   Improved prompt input with multi-line support and clearer instructions.
    *   Added a dedicated "Negative Prompt" input field.
*   **New Multi-Prompt Capability:** An "Enable Multi-Line Prompts" checkbox allows processing each line in the prompt box as a separate generation task sequentially.
*   **Fine-Grained Settings & Controls:**
    *   **Model Precision Selection:** Dropdown (`torch_dtype_dropdown`) to choose BF16 (Quality/Speed) or FP8 (Lower VRAM), triggering model reloads if changed.
    *   **Tiled VAE Control:** Checkbox (`tiled_vae_checkbox`) to enable/disable Tiled VAE for VRAM savings during decoding.
    *   **Explicit Dimensions:** Number inputs for `Width` and `Height` (must be divisible by 16).
    *   **Generation Variations:** Input (`num_generations_input`) to create multiple video variations (using different seeds) for *each* prompt.
    *   **Advanced Settings Accordion:** Grouped controls for `Sigma Shift`, `Denoising Strength`, `Output Quality (CRF)`, and `Save Metadata`.
    *   **Performance & VRAM Accordion:**
        *   VRAM Preset Helper dropdown linked to a textbox for fine-tuning persistent DiT parameters in VRAM.
        *   Specific inputs for VAE `Tile Size` (H/W) and `Tile Stride` (H/W).
*   **Improved Execution & Output:**
    *   Clearer button labels ("Generate Video(s)", "Cancel All", "Start Batch Process", "Open Outputs Folder").
    *   Dedicated "Cancel All" button (`cancel_btn`) for stopping ongoing tasks.
    *   "Open Outputs Folder" button (`open_folder_btn`) for quick access to the results directory.
    *   Output video display label clarified to "Last Generated Video Output".
*   **New Batch Processing UI:**
    *   Dedicated "Batch Processing" section with inputs for Input/Output folders.
    *   Checkboxes to control skipping existing files and using UI fallbacks for missing audio/prompts.
    *   "Start Batch Process" button.
*   **New Presets System:**
    *   Load/Save all UI settings via a dropdown and save button/textbox.
    *   Presets are stored as `.json` files in `./presets`.
    *   Automatically loads the *last used* preset on startup.
*   **New RIFE Frame Interpolation UI:**
    *   Dedicated section to configure optional RIFE post-processing.
    *   Select RIFE mode ("None", "2x FPS", "4x FPS").
    *   Set a maximum FPS limit for the RIFE output.

### Core Logic & Processing (`infer.py` vs. `old_infer.py`)

*   **Modular Argument Handling:** `infer.py` now receives all parameters as a dictionary from the app, removing internal `argparse` and improving reusability.
*   **Dynamic Model Loading:** Models are now loaded/reloaded by the app *only when necessary* (e.g., on first run or when precision/VRAM settings change). `infer.py`'s `load_models` function selects the correct Wan I2V model file (FP8/FP16) based on the requested precision.
*   **Robust Error Handling:** Significantly more `try...except` blocks throughout the app and inference script, providing better user feedback (`gr.Error`, `gr.Warning`) and detailed console logs (`traceback`).
*   **State Management & Cancellation:** The app manages generation state (`is_generating`, etc.) and passes a cancellation function (`cancel_fn`) down to the pipeline, allowing graceful interruption of the diffusion process.
*   **Precise Frame Calculation:** Correctly calculates the required number of frames (`4k+1` format) based on duration and FPS using a dedicated `calculate_frames` function.
*   **Full Batch Processing Logic:** The `process_batch` function in the app handles file discovery, fallback logic, multi-prompt processing within batches, and iterative calls to `infer.main`.
*   **Enhanced Logging:** `infer.py` provides much more detailed console output, including pretty-printed arguments, timings, tensor shapes, FFmpeg commands, and clear task identifiers.
*   **Improved Output Naming:** Centralized and more descriptive output file naming, incorporating sequential numbers, image stems (batch), prompt index (multi-prompt), and variation index. Input audio is copied to `./used_audios` with a matching name for reference.
*   **Direct FFmpeg Integration:** `infer.py` now directly calls `ffmpeg` via `subprocess` to merge video frames and audio, using optimized quality settings (`-crf`, `-preset`, etc.) and providing error details.
*   **New Metadata Saving:** Optionally creates a `.txt` file alongside the video containing a JSON dump of all settings, timings, filenames, RIFE status, etc., for reproducibility.
*   **First Frame Replacement:** Ensures the final video starts with the original input image by replacing the first generated frame *after* the diffusion process.
*   **New RIFE Integration:** `infer.py` can automatically invoke the `Practical-RIFE` script after FFmpeg merging if enabled in the UI, updating the final output path accordingly.

### Pipeline Enhancements (`wan_video.py` vs. `old_wan_video.py`)

*   **Integrated Cancellation Check:** The pipeline's `__call__` method now accepts and checks the `cancel_fn` at each denoising step, raising `CancelledError` if requested.
*   **Gradio Progress Reporting:** Accepts a `gradio_progress` object and updates it during the denoising loop, providing real-time progress feedback in the UI.

---

These changes collectively provide a more powerful, flexible, user-friendly, and stable application for generating talking head videos.
