# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 10:22:37 2026

@author: Yunche Zhu
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from data_loader_total import MetaSurfaceDataset

# --- 配置 ---
DATA_PATH = "E:/hkust/Meta_AI_Project/data_raw_total/" 
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_NAME = "forward_model_64.pth"

class ForwardCNN_64(nn.Module):
    def __init__(self):
        super(ForwardCNN_64, self).__init__()

        self.features = nn.Sequential(
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1000),
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x) 
        x = self.regressor(x)
        return x

def train():
    print(f" Check: Using {DEVICE} for Forward Model training")
    train_ds_full = MetaSurfaceDataset(DATA_PATH, target_size=64, augment=True)
    val_ds_full = MetaSurfaceDataset(DATA_PATH, target_size=64, augment=False)
    
    total_len = len(train_ds_full)
    indices = torch.randperm(total_len).tolist()
    split = int(0.8 * total_len) 
    
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = Subset(train_ds_full, train_indices)
    val_dataset = Subset(val_ds_full, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f" Loading: Train {len(train_dataset)} | Val {len(val_dataset)}")
    
    model = ForwardCNN_64().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(" Start Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, specs in train_loader: 
            imgs, specs = imgs.to(DEVICE), specs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, specs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, specs in val_loader:
                imgs, specs = imgs.to(DEVICE), specs.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, specs)
                val_loss += loss.item() * imgs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_NAME)
            print(f"    Save Best Model (Val Loss: {best_val_loss:.5f})")

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Forward Model Training Curve (Sigmoid Fixed)')
    plt.savefig('training_curve_forward.png')
    print(" Training Successful!")

if __name__ == "__main__":
    train()