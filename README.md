# Qwen3-TTS-GUI
An intuitive GUI app for Qwen3's TTS model

## Features

- **Train Voice**: Create voice clones from audio files or by recording
- **Use Voice**: Generate speech using trained voice models
- **Device Selection**: Choose between CUDA (GPU) or CPU for model inference
- **User-Friendly Interface**: Clean, tabbed interface built with tkinter

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```
2. Optionally install Flash-Attenuation:
```bash
pip install flash-attn --no-build-isolation
```
3. You might want [CUDA](https://developer.nvidia.com/cuda-downloads) as well (Nvidia GPUs only):
```plaintext
https://developer.nvidia.com/cuda-downloads
```
4. Install the [correct version of PyTorch according to your system](https://pytorch.org/get-started/locally/)
> [!IMPORTANT]  
> Make sure you select `PyTorch Build: Stable`, `Your OS: <Your OS>`, `Package: Pip`, `Language: Python`, and `Compute Platform: <CPU or The version of CUDA installed if using CUDA>`

## Usage

Run the GUI application:
```bash
python src/qwen_tts_gui.py
```

## How to Use

### Training a Voice

1. Go to the **Train Voice** tab
2. Enter a name for your voice
3. Select your preferred device (CUDA/CPU)
4. Choose training method:
   - **From Audio File + Transcript**: Browse for a WAV file and enter the transcript
   - **Record Audio**: Use the built-in recorder with a pre-made script
5. Click **Train Voice** to start training
6. The trained voice will be saved as `{voice_name}.pt`

### Using a Trained Voice

1. Go to the **Use Voice** tab
2. Browse and select a trained voice file (`.pt` file)
3. Select your preferred device (CUDA/CPU)
4. Enter the text you want to generate
5. Click **Generate Speech** to create the audio
6. The output will be saved as `{voice_name}_output.wav`

## Notes

- For CUDA/GPU usage, ensure you have CUDA-compatible PyTorch installed
- Flash Attention 2 will be used automatically if available, otherwise falls back to SDPA
- CPU mode uses float32 precision and eager attention implementation
- GPU mode uses bfloat16 precision for better performance


This readme and the comments in the code were created with the assistance of AI
