"""
Created on Sun Feb  1 11:37:55 2026
@author: Yunche Zhu
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from data_loader_total import MetaSurfaceDataset

DATA_PATH = "E:/hkust/Meta_AI_Project/data_raw_total/"
FORWARD_MODEL_PATH = "forward_model_64.pth"
SAVE_DIR = "results_wgan_64"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

BATCH_SIZE = 64
LR_G = 1e-4
LR_D = 1e-4
EPOCHS = 500  
Z_DIM = 100   
SPEC_DIM = 1000 
LAMBDA_GP = 10 


PHYSICS_WEIGHT = 1500 
BIN_WEIGHT = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ForwardCNN_64(nn.Module):
    def __init__(self):
        super(ForwardCNN_64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(512, 1000),
            nn.Sigmoid() 
        )
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.regressor(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(Z_DIM + SPEC_DIM, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), 
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1), 
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh() 
        )

    def forward(self, z, spectrum):
        x = torch.cat([z, spectrum], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        img = self.conv_blocks(x)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.img_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.spec_fc = nn.Sequential(
            nn.Linear(SPEC_DIM, 512),
            nn.LeakyReLU(0.2)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(256 * 4 * 4 + 512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, img, spectrum):
        img_feat = self.img_conv(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        spec_feat = self.spec_fc(spectrum)
        combined = torch.cat([img_feat, spec_feat], dim=1)
        score = self.final_fc(combined)
        return score


def compute_gradient_penalty(D, real_samples, fake_samples, spectrum):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(DEVICE)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, spectrum)
    
    fake = torch.ones(real_samples.size(0), 1).to(DEVICE)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train():
    print(" WGAN-GP Training Start (Refined)")
    print(f"   Physics Weight = {PHYSICS_WEIGHT}")
    print(f"   Binarization Weight = {BIN_WEIGHT}")

    dataset = MetaSurfaceDataset(DATA_PATH, target_size=64, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    Teacher = ForwardCNN_64().to(DEVICE)
    if os.path.exists(FORWARD_MODEL_PATH):
        Teacher.load_state_dict(torch.load(FORWARD_MODEL_PATH))
        print(" Physical Teacher Loaded")
    else:
        print(" Forward Model not found! Please train Teacher first.")
        return

    Teacher.eval() 
    for param in Teacher.parameters():
        param.requires_grad = False 
    
    optimizer_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.9))
    
    history = {'D_loss': [], 'G_loss': [], 'Physics_loss': [], 'Bin_loss': []}

    print(" Training Loop Start...")
    for epoch in range(EPOCHS):
        for i, (real_imgs, spectrum) in enumerate(dataloader):
            
            real_imgs = real_imgs.to(DEVICE)
            spectrum = spectrum.to(DEVICE)
            
            real_imgs = real_imgs * 2 - 1

            optimizer_D.zero_grad()
            
            z = torch.randn(BATCH_SIZE, Z_DIM).to(DEVICE)
            fake_imgs = G(z, spectrum) 
            
            real_validity = D(real_imgs, spectrum)
            fake_validity = D(fake_imgs.detach(), spectrum)
            
            gradient_penalty = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data, spectrum)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optimizer_D.step()

            if i % 2 == 0:
                optimizer_G.zero_grad()
                
                gen_imgs = G(z, spectrum)
                
                fake_validity = D(gen_imgs, spectrum)
                g_adv_loss = -torch.mean(fake_validity)
                
                gen_imgs_norm = (gen_imgs + 1) / 2
                pred_spectrum = Teacher(gen_imgs_norm)
                g_physics_loss = nn.MSELoss()(pred_spectrum, spectrum)

                g_bin_loss = torch.mean((1 - torch.abs(gen_imgs)) ** 2)
                
                g_loss = g_adv_loss + (PHYSICS_WEIGHT * g_physics_loss) + (BIN_WEIGHT * g_bin_loss)
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                optimizer_G.step()
                
                if i % 50 == 0:
                    print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}] [D: {d_loss.item():.4f}] [Phys: {g_physics_loss.item():.4f}] [Bin: {g_bin_loss.item():.4f}]")
                    history['D_loss'].append(d_loss.item())
                    history['G_loss'].append(g_adv_loss.item())
                    history['Physics_loss'].append(g_physics_loss.item())
                    history['Bin_loss'].append(g_bin_loss.item())

        if epoch % 10 == 0:
            save_img_tensor = (fake_imgs.data[:25] + 1) / 2
            save_image(save_img_tensor, f"{SAVE_DIR}/{epoch}.png", nrow=5, normalize=False)
            print(f" Saved image: {SAVE_DIR}/{epoch}.png")
            torch.save(G.state_dict(), f"{SAVE_DIR}/generator_epoch_{epoch}.pth")
            torch.save(G.state_dict(), f"{SAVE_DIR}/generator_latest.pth")

    print(" Training Complete!")

if __name__ == "__main__":
    train()