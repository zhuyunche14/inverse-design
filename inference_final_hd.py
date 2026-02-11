

#inference.py


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import cv2
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_PATH = "results_wgan_64/generator_latest.pth"
FWD_PATH = "forward_model_64.pth"
DATA_PATH = "E:/hkust/Meta_AI_Project/data_raw_total/"
UNIT_CELL_SIZE = 340 


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
            nn.Linear(512, 1000)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.regressor(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1100, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z, spectrum):
        x = torch.cat([z, spectrum], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        img = self.conv_blocks(x)
        return img


def smooth_image_hd(img_64_np, upscale_factor=8):
    img_uint8 = (img_64_np * 255).astype(np.uint8)
    target_size = (64 * upscale_factor, 64 * upscale_factor)
    img_high = cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_CUBIC)
    img_blur = cv2.GaussianBlur(img_high, (31, 31), 0)
    _, img_hd = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)
    return img_hd

def measure_universal(img_hd, unit_nm=340):

    contours, _ = cv2.findContours(img_hd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return "Empty", "gray"
    

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (cx, cy), (w_px, h_px), angle = rect
    

    scale = unit_nm / img_hd.shape[0]
    area = cv2.contourArea(c)
    box_area = w_px * h_px
    if box_area == 0: return "Error", "gray"
    

    solidity = area / box_area 
    

    center_y, center_x = int(cy), int(cx)
    y1, y2 = max(0, center_y-4), min(img_hd.shape[0], center_y+4)
    x1, x2 = max(0, center_x-4), min(img_hd.shape[1], center_x+4)
    center_region = img_hd[y1:y2, x1:x2]
    is_center_filled = np.mean(center_region) > 127
    

    if w_px < h_px: w_px, h_px = h_px, w_px; angle += 90
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    
    L_nm = w_px * scale
    W_nm = h_px * scale


    
    if solidity > 0.85:

        info = f"[Rect]\nL: {L_nm:.0f}\nW: {W_nm:.0f}\nAng: {angle:.0f}°"
        color = "#00CC66" 
        
    elif is_center_filled and solidity < 0.85:
        info = f"[Cross]\nL: {L_nm:.0f}\nW: {W_nm:.0f}\nAng: {angle:.0f}°"
        color = "#3399FF" 
        
    elif not is_center_filled and solidity > 0.3:
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: perimeter = 1
        arm_width = area / (perimeter/2) * scale 
        info = f"[SRR]\nSpan: {L_nm:.0f}\nArm: ~{arm_width:.0f}\nAng: {angle:.0f}°"
        color = "#FF0055" 
        
    else:

        info = f"[Needle]\nBox: {L_nm:.0f}x{W_nm:.0f}\nFreeform"
        color = "#9900CC" 
        
    return info, color

def main():
    print(" (Rect/Cross/SRR/Needle)...")
    
    G = Generator().to(DEVICE)
    if os.path.exists(GEN_PATH):
        G.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
        print("sucessful")
    else:
        print(f"fault : {GEN_PATH}")
        return
    G.eval()

    Teacher = ForwardCNN_64().to(DEVICE)
    if os.path.exists(FWD_PATH):
        Teacher.load_state_dict(torch.load(FWD_PATH, map_location=DEVICE))
        print("physical")
    else:
        print("fault")
        return
    Teacher.eval()

    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.mat')]
    random.shuffle(files)
    
    target_file = None
    real_spec = None
    
    print("rondomly...")
    for f in files:
        try:
            data = sio.loadmat(os.path.join(DATA_PATH, f))
            spec = data['T_val'].flatten() if 'T_val' in data else data['spectrum'].flatten()
            if not np.isnan(spec).any(): 
                target_file = f
                real_spec = spec
                break
        except: continue
    
    if real_spec is None:
        print("fault")
        return

    num_designs = 4
    target_tensor = torch.from_numpy(real_spec).float().unsqueeze(0).to(DEVICE)
    target_batch = target_tensor.repeat(num_designs, 1)
    z = torch.randn(num_designs, 100).to(DEVICE)

    with torch.no_grad():
        gen_imgs = G(z, target_batch) 
        gen_imgs_norm = (gen_imgs + 1) / 2
        pred_spectra = Teacher(gen_imgs_norm)

    human_target = np.clip(np.abs(real_spec), 0, 1)
    human_preds = np.clip(np.abs(pred_spectra.cpu().numpy()), 0, 1)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 4)

    ax_spec = fig.add_subplot(gs[:, :2])
    x_axis = np.linspace(400, 800, 1000)
    ax_spec.plot(x_axis, human_target, 'k--', linewidth=3, label='Target')
    
    line_colors = ['#FF0055', '#00CC66', '#3399FF', '#9900CC']
    
    for i in range(num_designs):
        ax_spec.plot(x_axis, human_preds[i], color=line_colors[i], linewidth=2, alpha=0.8, label=f'Design {i+1}')
    
    ax_spec.set_title(f"Target: {target_file}", fontsize=12)
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylim([-0.05, 1.05])
    ax_spec.legend()
    ax_spec.grid(True, linestyle=':', alpha=0.6)

    img_positions = [gs[0, 2], gs[0, 3], gs[1, 2], gs[1, 3]]
    for i in range(num_designs):
        ax_img = fig.add_subplot(img_positions[i])
        raw_img_np = gen_imgs_norm[i].squeeze().cpu().numpy()
        
        hd_img = smooth_image_hd(raw_img_np, upscale_factor=8)
        
        info_str, type_color = measure_universal(hd_img, unit_nm=UNIT_CELL_SIZE)
        
        ax_img.imshow(hd_img, cmap='gray', vmin=0, vmax=255)
        
        ax_img.set_title(info_str, color=type_color, fontweight='bold', fontsize=10, 
                         backgroundcolor='white', y=0.98)
        
        ax_img.axis('off')
        for spine in ax_img.spines.values():
            spine.set_edgecolor(type_color)
            spine.set_linewidth(3)

    plt.tight_layout()
    save_path = "final_result_universal.png"
    plt.savefig(save_path, dpi=300)
    print(f"check: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()