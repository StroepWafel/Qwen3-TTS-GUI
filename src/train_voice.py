import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel

SAMPLE_RATE = 44100
CHANNELS = 1

voice_name = input("Please enter the name of the voice to create:\n")

print(f"Creating voice '{voice_name}'")

ref_audio = "mic.wav"
ref_text  = "Quick brown foxes jump over lazy dogs while musicians play jazzy tunes in the quiet park."

# Default input device
device_index = sd.default.device[0]
print("Using default input device:", sd.query_devices(device_index)["name"])

print("Please read the following text aloud (press enter when ready): \n" + ref_text)

input()

print("Recording... press Enter to stop")

# List to hold chunks
audio_chunks = []

def callback(indata, frames, time, status):
    audio_chunks.append(indata.copy())

# Open stream
with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
    input()  # Wait for user to press Enter

# Combine chunks into a single array
audio_data = np.concatenate(audio_chunks, axis=0)

# Save to file
sf.write("mic.wav", audio_data, SAMPLE_RATE)
print("Saved mic.wav")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

ref_audio = "mic.wav"
ref_text  = "Quick brown foxes jump over lazy dogs while musicians play jazzy tunes in the quiet park."

# Generate a "voice clone prompt" — this holds the speaker embedding
prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,
)

# Save the prompt to disk
torch.save(prompt_items, f"{voice_name}.pt")
