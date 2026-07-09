import os
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import io
from pathlib import Path
from train import CNNBiLSTM, AMSoftmax
import uvicorn
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("c:/Users/Lenovo/Downloads/Project/models")

# Initialize Model
model = CNNBiLSTM(embedding_dim=128).to(device)
am_softmax = AMSoftmax(in_features=128, n_classes=2).to(device)

model_path = MODELS_DIR / "best_deepfake_detector.pth"
if model_path.exists():
    checkpoint = torch.load("models/best_deepfake_detector.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    am_softmax.load_state_dict(checkpoint['am_softmax_state_dict'])
else:
    print(f"Warning: Model checkpoint not found at {model_path}. Using untrained weights.")
    
model.eval()
am_softmax.eval()

SAMPLE_RATE = 16000
WINDOW_SAMPLES = int(SAMPLE_RATE * 2.0)

lfcc_transform = T.LFCC(
    sample_rate=SAMPLE_RATE,
    n_lfcc=40,
    speckwargs={"n_fft": 512, "hop_length": 160, "center": False}
)

def process_audio_tensor(waveform, sr):
    if sr != SAMPLE_RATE:
        resample = T.Resample(sr, SAMPLE_RATE)
        waveform = resample(waveform)
    
    if waveform.shape[-1] > WINDOW_SAMPLES:
        # Extract center 2 seconds to avoid leading silence in TTS
        start = (waveform.shape[-1] - WINDOW_SAMPLES) // 2
        waveform = waveform[:, start:start + WINDOW_SAMPLES]
    elif waveform.shape[-1] < WINDOW_SAMPLES:
        pad_len = WINDOW_SAMPLES - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        
    features = lfcc_transform(waveform)
    return features.unsqueeze(0) # Add batch dimension

def get_inference(features, waveform=None):
    with torch.no_grad():
        features = features.to(device)
        embeddings = model(features)
        
        W_norm = torch.nn.functional.normalize(am_softmax.W, p=2, dim=1)
        x_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        cosine = torch.nn.functional.linear(x_norm, W_norm)
        
        probs = torch.nn.functional.softmax(cosine * 10.0, dim=1)
        prob_synthetic = probs[0][1].item()
        
        hf_ratio = 0.0
        if waveform is not None:
            wf = waveform[0].cpu()
            fft = torch.fft.rfft(wf)
            mag = torch.abs(fft)
            mid = len(mag) // 2
            hf_ratio = (torch.sum(mag[mid:]) / (torch.sum(mag[:mid]) + 1e-8)).item()
                
    return prob_synthetic, hf_ratio

@app.post("/api/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    import soundfile as sf
    import io
    contents = await file.read()
    audio_data, sr = sf.read(io.BytesIO(contents))
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=1)
    waveform = torch.tensor(audio_data, dtype=torch.float32).transpose(0, 1)
    
    features = process_audio_tensor(waveform, sr)
    prob_synthetic, hf_ratio = get_inference(features, waveform)
    
    score = prob_synthetic * 100
    classification = "Synthetic" if score > 65 else "Authentic"
    
    return {
        "authenticity_score": score,
        "classification": classification,
        "metadata": {
            "source": "File Upload",
            "bandwidth": f"{sr} Hz",
            "hf_ratio": f"{hf_ratio:.3f}"
        }
    }

@app.websocket("/ws/live-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = b""
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer += data
            if len(buffer) > 16000 * 2 * 2: 
                audio_np = np.frombuffer(buffer[:16000*2*2], dtype=np.int16).astype(np.float32) / 32768.0
                waveform = torch.tensor(audio_np).unsqueeze(0)
                
                features = process_audio_tensor(waveform, 16000)
                prob_synthetic, hf_ratio = get_inference(features, waveform)
                
                score = prob_synthetic * 100
                classification = "Synthetic" if score > 65 else "Authentic"
                
                await websocket.send_text(json.dumps({
                    "authenticity_score": score,
                    "classification": classification,
                    "status": "active",
                    "metadata": {
                        "source": "WebSocket Stream",
                        "bandwidth": "16000 Hz",
                        "hf_ratio": f"{hf_ratio:.3f}"
                    }
                }))
                buffer = buffer[16000*2*2:] 
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5003)
