import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import os

PROCESSED_DIR = Path("c:/Users/Lenovo/Downloads/Project/processed_data")
MODELS_DIR = Path("c:/Users/Lenovo/Downloads/Project/models")

# AM-Softmax Loss (Additive Margin Softmax)
class AMSoftmax(nn.Module):
    def __init__(self, in_features, n_classes=2, m=0.35, s=30.0):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_features = in_features
        self.W = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, label):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        W_norm = torch.nn.functional.normalize(self.W, p=2, dim=1)
        
        cosine = torch.nn.functional.linear(x_norm, W_norm)
        
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (cosine - one_hot * self.m) * self.s
        return output

# CNN-BiLSTM Architecture
class CNNBiLSTM(nn.Module):
    def __init__(self, input_shape=(40, 201), embedding_dim=256):
        super(CNNBiLSTM, self).__init__()
        # Input shape: (Batch, Channels, Freq, Time)
        # For LFCC, we have (Batch, 1, n_lfcc, Time)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, None)) # Pool freq axis to 10
        )
        
        self.lstm = nn.LSTM(
            input_size=64 * 10,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        # x is already (Batch, Channels, Freq, Time)
        x = self.cnn(x) # (Batch, 64, 10, Time')
        
        # Reshape for LSTM: (Batch, Time', 64 * 10)
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, c, f = x.size()
        x = x.reshape(batch_size, time_steps, c * f)
        
        out, _ = self.lstm(x) # out: (Batch, Time', 256)
        
        # Take last time step
        out = out[:, -1, :] # (Batch, 256)
        embeddings = self.fc(out)
        return embeddings

def main():
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    
    print("Loading datasets...")
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = CNNBiLSTM(embedding_dim=128).to(device)
    am_softmax = AMSoftmax(in_features=128, n_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer for both model and AMSoftmax weights
    optimizer = optim.Adam(
        list(model.parameters()) + list(am_softmax.parameters()),
        lr=1e-3, weight_decay=1e-5
    )
    
    epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        am_softmax.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(inputs)
            outputs = am_softmax(embeddings, labels)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        model.eval()
        am_softmax.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                embeddings = model(inputs)
                outputs = am_softmax(embeddings, labels)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'am_softmax_state_dict': am_softmax.state_dict(),
            }, MODELS_DIR / "best_deepfake_detector.pth")
            print("=> Saved Best Model!")

if __name__ == "__main__":
    main()
