import os
import torch
import soundfile as sf
import numpy as np
import glob

def get_hf_ratio(waveform):
    wf = waveform[0]
    fft = torch.fft.rfft(wf)
    mag = torch.abs(fft)
    mid = len(mag) // 2
    hf_ratio = (torch.sum(mag[mid:]) / (torch.sum(mag[:mid]) + 1e-8)).item()
    return hf_ratio

root_dir1 = r"c:\Users\Lenovo\Downloads\Project\ljspeech_dataset\authentic"
root_dir2 = r"c:\Users\Lenovo\Downloads\Project\yesno\waves_yesno"

print(f"Evaluating HF Ratio across authentic audio...")

high_ratio_count = 0
total_count = 0
files = glob.glob(os.path.join(root_dir1, "*.wav")) + glob.glob(os.path.join(root_dir2, "*.wav"))

for f in files[:100]: # test 100 files
    total_count += 1
    audio_data, sr = sf.read(f)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
    hf_ratio = get_hf_ratio(waveform)
    
    if hf_ratio > 0.05:
        print(f"HIGH HF for authentic: {os.path.basename(f)} - Ratio: {hf_ratio:.4f}")
        high_ratio_count += 1

print(f"Summary: {high_ratio_count}/{total_count} authentic files have HF > 0.05.")
