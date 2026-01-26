import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

voice_name = input("Please enter the name of the voice to use:\n")

print(f"Using voice '{voice_name}'")

text_prompt = input("Please enter the text to generate:\n")

print(f"Generating text '{text_prompt}'")


model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# Allowlist the class
torch.serialization.add_safe_globals([VoiceClonePromptItem])

# Load the saved voice clone prompt
prompt_items = torch.load(f"{voice_name}.pt")

# Generate speech
wavs, sr = model.generate_voice_clone(
    text=text_prompt,
    language="English",
    voice_clone_prompt=prompt_items,
)

sf.write(f"{voice_name}.wav", wavs[0], sr)
