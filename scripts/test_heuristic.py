import torch
import soundfile as sf
import numpy as np
import glob

def get_hf_ratio(file_path):
    y, sr = sf.read(file_path)
    if y.ndim == 2: y = y[:, 0]
    wf = torch.tensor(y, dtype=torch.float32)
    
    fft = torch.fft.rfft(wf)
    mag = torch.abs(fft)
    mid = len(mag) // 2
    
    hf_energy = torch.sum(mag[mid:])
    lf_energy = torch.sum(mag[:mid])
    hf_ratio = (hf_energy / (lf_energy + 1e-8)).item()
    print(f"{file_path}: HF Ratio = {hf_ratio:.4f}")

for f in glob.glob("ljspeech_dataset/authentic/*.wav")[:2]:
    get_hf_ratio(f)
for f in glob.glob("generated_audio/**/*.wav", recursive=True)[:2]:
    get_hf_ratio(f)
get_hf_ratio("trumpet.wav")
