import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from train import CNNBiLSTM, AMSoftmax
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import librosa

device = torch.device("cpu")
model = CNNBiLSTM(embedding_dim=128)
am = AMSoftmax(in_features=128, n_classes=2)

checkpoint = torch.load("models/best_deepfake_detector.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
am.load_state_dict(checkpoint['am_softmax_state_dict'])

model.eval()
am.eval()

lfcc = T.LFCC(
    sample_rate=16000,
    n_lfcc=40,
    speckwargs={"n_fft": 512, "hop_length": 160, "center": False}
)

def infer(file_path):
    y, sr = sf.read(file_path)
    if y.ndim == 1: y = np.expand_dims(y, axis=1)
    wf = torch.tensor(y, dtype=torch.float32).transpose(0, 1)
    
    if sr != 16000:
        wf = T.Resample(sr, 16000)(wf)
        
    if wf.shape[-1] > 32000:
        start = (wf.shape[-1] - 32000) // 2
        wf = wf[:, start:start+32000]
    if wf.shape[-1] < 32000:
        wf = torch.nn.functional.pad(wf, (0, 32000 - wf.shape[-1]))
        
    feat = lfcc(wf).unsqueeze(0)
    
    with torch.no_grad():
        emb = model(feat)
        W_norm = torch.nn.functional.normalize(am.W, p=2, dim=1)
        x_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
        cosine = torch.nn.functional.linear(x_norm, W_norm)
        probs = torch.nn.functional.softmax(cosine * 10.0, dim=1)
        print(f"{file_path}: Authentic={probs[0][0].item():.4f}, Synthetic={probs[0][1].item():.4f}")

import glob
for f in glob.glob("ljspeech_dataset/authentic/*.wav")[:3]:
    infer(f)
    
for f in glob.glob("generated_audio/**/*.wav", recursive=True)[:3]:
    infer(f)

# Also test a raw librosa chunk directly
y, sr = librosa.load(librosa.example('trumpet'), sr=16000)
sf.write("trumpet.wav", y, sr)
infer("trumpet.wav")
