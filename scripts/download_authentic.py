import os
import soundfile as sf
from datasets import load_dataset
from pathlib import Path

def download_auth():
    out_dir = Path("ljspeech_dataset/authentic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading 400 authentic LJSpeech samples via streaming...")
    ds = load_dataset("lj_speech", split="train", streaming=True)
    
    count = 0
    for item in ds:
        audio = item["audio"]
        sf.write(str(out_dir / f"auth_{count}.wav"), audio["array"], audio["sampling_rate"])
        count += 1
        if count % 50 == 0:
            print(f"Downloaded {count} samples...")
        if count >= 400:
            break
    print("Finished downloading authentic data.")

if __name__ == "__main__":
    download_auth()
