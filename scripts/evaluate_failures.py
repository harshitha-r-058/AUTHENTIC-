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

root_dir = r"c:\Users\Lenovo\Downloads\Project\generated_audio"
directories = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

print(f"Evaluating HF Ratio across {len(directories)} directories...")

low_ratio_count = 0
total_count = 0

for d in directories:
    files = glob.glob(os.path.join(d, "*.wav"))
    for f in files[:10]: # test 10 files per dir
        total_count += 1
        audio_data, sr = sf.read(f)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        hf_ratio = get_hf_ratio(waveform)
        
        if hf_ratio <= 0.3:
            print(f"FAILED (HF <= 0.3): {os.path.basename(d)}/{os.path.basename(f)} - Ratio: {hf_ratio:.4f}")
            low_ratio_count += 1

print(f"Summary: {low_ratio_count}/{total_count} files bypassed the DSP HF > 0.3 heuristic.")
