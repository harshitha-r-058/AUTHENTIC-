import os
import glob
import math
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from pathlib import Path

# Parameters
SAMPLE_RATE = 16000
WINDOW_DURATION = 2.0  # 2.0 seconds
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
DATA_DIR = Path("c:/Users/Lenovo/Downloads/Project")
GENERATED_DIR = DATA_DIR / "generated_audio"
PROCESSED_DIR = DATA_DIR / "processed_data"

def pad_or_trim(audio, target_length):
    if audio.shape[-1] > target_length:
        max_start = audio.shape[-1] - target_length
        start = np.random.randint(0, max_start + 1)
        return audio[:, start:start+target_length]
    elif audio.shape[-1] < target_length:
        pad_len = target_length - audio.shape[-1]
        return torch.nn.functional.pad(audio, (0, pad_len))
    return audio

def telephony_augmentation(waveform, sample_rate):
    # Simulate GSM telephony channel
    effects = [
        ["sinc", "-a", "120", "300-3400"]
    ]
    try:
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
    except AttributeError:
        # sox_effects not available on this platform (e.g., Windows), skip bandpass
        pass
        
    # Add noise
    snr_db = torch.randint(0, 30, (1,)).item()
    snr = 10 ** (snr_db / 20)
    noise = torch.randn_like(waveform)
    signal_rms = waveform.norm(p=2) / math.sqrt(waveform.numel())
    noise_rms = noise.norm(p=2) / math.sqrt(noise.numel())
    scale = signal_rms / (snr * noise_rms)
    noisy_waveform = (waveform + scale * noise) / 2
    return noisy_waveform

def extract_features(waveform, sample_rate):
    # LFCC Feature Extraction
    lfcc_transform = T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=40,
        speckwargs={"n_fft": 512, "hop_length": 160, "center": False}
    )
    lfcc = lfcc_transform(waveform)
    return lfcc

def process_file(filepath, label):
    import soundfile as sf
    audio_data, sr = sf.read(filepath)
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=1)
    waveform = torch.tensor(audio_data, dtype=torch.float32).transpose(0, 1)
    # Resample
    if sr != SAMPLE_RATE:
        resample = T.Resample(sr, SAMPLE_RATE)
        waveform = resample(waveform)
    
    # Trim/Pad
    waveform = pad_or_trim(waveform, WINDOW_SAMPLES)
    
    # Augmentation
    waveform_aug = telephony_augmentation(waveform, SAMPLE_RATE)
    
    # Feature extraction
    features = extract_features(waveform_aug, SAMPLE_RATE)
    return features.numpy(), label

def main():
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    
    data = []
    labels = []
    
    print("Processing Authentic Samples...")
    authentic_dir = DATA_DIR / "ljspeech_dataset" / "authentic"
    authentic_files = list(authentic_dir.glob("*.wav"))
    
    yesno_dir = DATA_DIR / "yesno" / "waves_yesno"
    if yesno_dir.exists():
        authentic_files += list(yesno_dir.glob("*.wav"))
    
    if len(authentic_files) == 0:
        raise RuntimeError("No authentic files found.")
        
    authentic_samples = min(300, len(authentic_files))
    for i, filepath in enumerate(authentic_files[:authentic_samples]):
        feat, label = process_file(str(filepath), 0) # 0 for Authentic
        data.append(feat)
        labels.append(0)
        
    print(f"Processed {authentic_samples} authentic samples.")
            
    print("Processing Synthetic Samples...")
    synthetic_files = list(GENERATED_DIR.glob("**/*.wav"))
    np.random.shuffle(synthetic_files)
    
    synthetic_samples = authentic_samples # Balance the dataset
    for i, filepath in enumerate(synthetic_files[:synthetic_samples]):
        feat, label = process_file(str(filepath), 1) # 1 for Synthetic
        data.append(feat)
        labels.append(label)
        
    print(f"Processed {synthetic_samples} synthetic samples.")
        
    X = np.array(data)
    y = np.array(labels)
    
    print(f"Total samples processed: {len(X)}")
    print(f"Feature shape: {X.shape}")
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print("Saving processed tensors...")
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    print("Preprocessing completed.")

if __name__ == "__main__":
    main()
