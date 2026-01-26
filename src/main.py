import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel

SAMPLE_RATE = 44100
CHANNELS = 1

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
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

wavs, sr = model.generate_voice_clone(
    text="This audio is a test of how good of an AI this is and also whether it can save voices",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)

# Extract embedding
speaker_emb = model.extract_speaker_embedding(ref_audio, ref_text)

# Save embedding to disk
torch.save(speaker_emb, "my_cloned_voice.pt")