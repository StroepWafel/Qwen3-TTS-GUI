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

## Usage

Run the GUI application:
```bash
python src/qwen_tts_gui.py
```

## Building an Executable

To create a standalone `.exe` file (Windows) or executable (Linux/Mac):

### Prerequisites
1. Install PyInstaller:
```bash
pip install pyinstaller
```

2. Install all dependencies:
```bash
pip install -r requirements.txt
```

### Building

**Windows:**
```bash
build_exe.bat
```

**Linux/Mac:**
```bash
chmod +x build_exe.sh
./build_exe.sh
```

**Or manually:**
```bash
python -m PyInstaller qwen_tts_gui.spec
```

**Note:** If you get an error that `pyinstaller` is not recognized, use `python -m PyInstaller` instead.

The executable will be created in the `dist` folder.

### Notes for Building
- The build process may take several minutes due to the large PyTorch dependencies
- The resulting executable will be large (several GB) due to PyTorch and model dependencies
- Ensure you have enough disk space (at least 5-10 GB free)
- The first model download will happen when you first run the executable

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