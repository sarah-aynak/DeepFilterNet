import os
import soundfile as sf
import numpy as np

# Path where all your enhanced audio files are saved
enhanced_dir = "results/"
output_path = "enhanced_all_combined.wav"

# Collect files (sorted for consistent order)
files = sorted([f for f in os.listdir(enhanced_dir) if f.endswith(".wav")])

all_audio = []

for f in files:
    audio, sr = sf.read(os.path.join(enhanced_dir, f))
    all_audio.append(audio)

# Concatenate into one long audio array
final_audio = np.concatenate(all_audio, axis=0)

# Save the combined wav
sf.write(output_path, final_audio, sr)

print(f"Saved combined file: {output_path}")
