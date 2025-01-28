import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

import numpy as np
import cv2
from PIL import Image

# =======================================
# 1. CONFIG / HYPERPARAMETERS
# =======================================
CONFIG = {
    "image_path": "input.jpg",      # Image file to load
    "target_class": 1,           # None => pick top predicted
    "num_iters": 50,               # Number of gradient steps
    "lr": 0.01,                     # Learning rate (lower than 0.1)
    "K": 20,                        # Number of masks in the schedule
    "gaussian_kernel_size": 17,      # For smoothing in SmoothAndPool
    "blur_ksize": 51,              # For create_blurred_background
    "temp_factor": 1,            # Temperature factor for exponentiate
    "tv_lambda": 0.01,            # Weight for TV penalty
}

# =======================================
# 2. SETUP & UTILITIES
# =======================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load a pre-trained VGG16
model = models.vgg16(pretrained=True).to(device)
model.eval()

# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((224, 224)),  # Force 224x224
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])

inv_transform = T.Compose([
    T.Normalize(mean=[0,0,0], std=[1/s for s in imagenet_std]),
    T.Normalize(mean=[-m for m in imagenet_mean], std=[1,1,1]),
])

def load_image(image_path):
    """
    Load image, resize to (224,224), normalize => shape [1,3,224,224].
    """
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    print(f"Loaded image: {image_path} => {tuple(tensor.shape)}")
    return tensor

def create_blurred_background(img, ksize=51):
    """
    Single Gaussian blur with OpenCV. shape [B,3,H,W].
    """
    img_np = img.detach().cpu().numpy()  # (B,3,H,W)
    out_np = np.zeros_like(img_np)
    for b in range(img_np.shape[0]):
        for c in range(img_np.shape[1]):
            out_np[b,c] = cv2.GaussianBlur(img_np[b,c], (ksize, ksize), sigmaX=0)
    return torch.from_numpy(out_np).to(device)

def apply_mask(x, mask, blur_image):
    """
    x, blur_image: [B,3,H,W]
    mask: [B,1,H,W] => broadcast to 3ch => combine
    """
    mask_3ch = mask.repeat(1, x.size(1), 1, 1)
    return mask_3ch * x + (1 - mask_3ch) * blur_image

# =======================================
# 3. SMOOTH & POOL MODULE
# =======================================
class SmoothAndPool(nn.Module):
    """
    1) Gaussian smoothing (5x5 => same size)
    2) 'Soft' max-like pooling (3x3 => same size)
    With a reduced temperature factor from outside config.
    """
    def __init__(self, kernel_size=5, temp_factor=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.temp_factor = temp_factor
        sigma = kernel_size / 6.0

        # Build a 2D Gaussian kernel
        ax = torch.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)
        xx = ax.repeat(kernel_size,1)
        yy = xx.t()
        kernel = torch.exp(-(xx**2 + yy**2)/(2.*sigma*sigma))
        kernel = kernel / kernel.sum()

        # Convolution
        self.gaussian_filter = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=False
        )
        with torch.no_grad():
            self.gaussian_filter.weight[:] = kernel.view(1,1,kernel_size,kernel_size)

    def forward(self, m):
        # 1) Gaussian
        m_smooth = self.gaussian_filter(m)  # same [B,1,H,W]

        # 2) Soft 'max-like' pooling
        exp_x = torch.exp(m_smooth * self.temp_factor)  
        pool = F.avg_pool2d(exp_x, kernel_size=3, stride=1, padding=1)
        denom= F.avg_pool2d(torch.ones_like(exp_x), kernel_size=3, stride=1, padding=1)
        return pool / (denom + 1e-7)

# =======================================
# 4. BUILD A 'SCHEDULE' OF MASKS
# =======================================
def mask_with_area(em, area_frac):
    """
    Solve t s.t. mean(sigmoid(em - t))=area_frac (bisection).
    """
    lower, upper = -5.0, 5.0
    for _ in range(20):
        mid = 0.5*(lower + upper)
        mk = torch.sigmoid(em - mid)
        frac_now = mk.mean().item()
        if frac_now > area_frac:
            lower = mid
        else:
            upper = mid
    final_t = 0.5*(lower + upper)
    mk = torch.sigmoid(em - final_t)
    return mk

def generate_mask_schedule(em, K=20):
    """
    Return list of (K+1) masks from fraction=0..1.
    """
    masks = []
    for i in range(K+1):
        frac = i/float(K)
        mk = mask_with_area(em, frac)
        masks.append(mk)
    return masks

# =======================================
# 5. MAIN SALIENCY-AS-SCHEDULE
# =======================================
def saliency_as_schedule(
    image_t,
    target_class=None,
    lr=0.05,
    num_iters=300,
    K=20,
    gaussian_kernel_size=5,
    blur_ksize=51,
    temp_factor=2.0,
    tv_lambda=0.001
):
    """
    image_t: [1,3,224,224]
    target_class: class idx or None => pick top
    lr: learning rate
    num_iters: # of steps
    K: # of masks in schedule
    gaussian_kernel_size: for SmoothAndPool
    blur_ksize: for background blur
    temp_factor: how strongly to exponentiate in soft pooling
    tv_lambda: weight for total-variation penalty
    """
    # 1) Blur
    blur_im = create_blurred_background(image_t, ksize=blur_ksize)
    print("image_t shape:", image_t.shape)
    print("blur_im shape:", blur_im.shape)

    # 2) target_class
    with torch.no_grad():
        logits = model(image_t)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        print(f"Target class = {target_class}, initial logit={logits[0,target_class]:.4f}")

    # 3) pre-mask param
    B, C, H, W = image_t.shape
    m_param = torch.zeros(B, 1, H, W, device=device, requires_grad=True)
    print("m_param shape:", m_param.shape)

    # 4) smoothing module
    sp_module = SmoothAndPool(kernel_size=gaussian_kernel_size, temp_factor=temp_factor).to(device)

    # 5) optimizer
    optimizer = torch.optim.Adam([m_param], lr=lr)

    for it in range(num_iters):
        optimizer.zero_grad()

        # (a) e_m => sp_module
        e_m = sp_module(m_param)  # [1,1,224,224]

        # (b) build (K+1) masks
        masks = generate_mask_schedule(e_m, K=K)

        # (c) insertion+deletion
        insertion_loss = 0.0
        deletion_loss  = 0.0
        for mk in masks:
            # insertion
            ins_im = apply_mask(image_t, mk, blur_im)
            ins_logit = model(ins_im)[:, target_class]
            insertion_loss -= ins_logit

            # deletion
            del_im = apply_mask(image_t, 1 - mk, blur_im)
            del_logit = model(del_im)[:, target_class]
            deletion_loss += del_logit

        # total-variation penalty to avoid single-pixel spikes
        tv_loss = (torch.abs(m_param[:,:,1:,:] - m_param[:,:,:-1,:]).sum() +
                   torch.abs(m_param[:,:,:,1:] - m_param[:,:,:,:-1]).sum())

        total_loss = (insertion_loss + deletion_loss) + tv_lambda * tv_loss

        total_loss.backward()
        optimizer.step()

        if (it+1) % 50 == 0:
            print(f"[{it+1}/{num_iters}] total_loss={total_loss.item():.4f} "
                  f"(ins={insertion_loss.item():.4f}, del={deletion_loss.item():.4f}, tv={tv_loss.item():.2f})")

    # 6) final saliency => [0..1]
    with torch.no_grad():
        e_m = sp_module(m_param)
        saliency = e_m - e_m.min()
        saliency = saliency / (saliency.max() + 1e-7)

    return saliency

# =======================================
# 6. DEMO / MAIN
# =======================================
if __name__ == "__main__":
    # Read config
    image_path        = CONFIG["image_path"]
    target_class      = CONFIG["target_class"]
    num_iters         = CONFIG["num_iters"]
    lr                = CONFIG["lr"]
    K                 = CONFIG["K"]
    gaussian_kernsize = CONFIG["gaussian_kernel_size"]
    blur_ksize        = CONFIG["blur_ksize"]
    temp_factor       = CONFIG["temp_factor"]
    tv_lambda         = CONFIG["tv_lambda"]

    # 1) Load
    image_t = load_image(image_path)

    # 2) Run SaS w/ modifications to avoid single pixel collapse
    sal_map = saliency_as_schedule(
        image_t=image_t,
        target_class=target_class,
        lr=lr,
        num_iters=num_iters,
        K=K,
        gaussian_kernel_size=gaussian_kernsize,
        blur_ksize=blur_ksize,
        temp_factor=temp_factor,
        tv_lambda=tv_lambda
    )

    # 3) Save saliency
    sal_np = sal_map.squeeze().cpu().numpy()  # [224,224], [0,1]
    sal_img = (sal_np * 255).astype(np.uint8)
    cv2.imwrite("sas_saliency.jpg", sal_img)
    print("Saved saliency to sas_saliency.jpg")

    # 4) Overlay
    inv_img = inv_transform(image_t[0]).clamp(0,1).permute(1,2,0).cpu().numpy()
    inv_img = (inv_img*255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sal_img, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(inv_img, 0.5, heatmap, 0.5, 0)
    cv2.imwrite("sas_overlay.jpg", overlay)
    print("Saved overlay to sas_overlay.jpg")
