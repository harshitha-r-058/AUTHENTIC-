import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def prepare():
    out_dir = Path("ljspeech_dataset/authentic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear old files
    for f in out_dir.glob("*.wav"):
        f.unlink()

    print("Fetching clean authentic speech using Librosa...")
    files = []
    
    # In newer librosa, we can use built-in small speech files
    try:
        files.append(librosa.example('libri1'))
    except:
        pass
    try:
        files.append(librosa.example('libri2'))
    except:
        pass
    try:
        files.append(librosa.example('libri3'))
    except:
        pass
        
    if not files:
        files.append(librosa.util.example_audio_file()) # Fallback

    count = 0
    for f in files:
        y, sr = librosa.load(f, sr=16000)
        chunk_size = sr * 3 # 3-second chunks!
        hop = int(sr * 0.5) # 0.5s hop for overlapping
        for i in range(0, len(y) - chunk_size, hop):
            chunk = y[i:i+chunk_size]
            sf.write(str(out_dir / f"auth_{count}.wav"), chunk, sr)
            count += 1
            if count >= 300:
                break
        if count >= 300:
            break

    print(f"Generated {count} continuous 3-second authentic samples.")

if __name__ == "__main__":
    prepare()
