import os
import time
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm

# ---------------------------
# Download images from Pexels
# ---------------------------

def download_pexels_images(query, total_images, per_page=20, save_dir="data/highRes", api_key_env="PEXELS_API_KEY", max_retries=3, delay=1.0):
    load_dotenv()
    API_KEY = os.getenv(api_key_env)
    if not API_KEY:
        raise ValueError(f"API key not found in environment variable '{api_key_env}'")
    
    os.makedirs(save_dir, exist_ok=True)
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": API_KEY}

    image_counter = 0
    page = 1
    downloaded_urls = set()

    print(f"Downloading {total_images} '{query}' images from Pexels...")

    while image_counter < total_images:
        params = {"query": query, "per_page": per_page, "page": page}
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(delay)
        else:
            print("Max retries reached, stopping downloads.")
            break

        photos = data.get("photos", [])
        if not photos:
            print("No more photos found.")
            break

        for photo in photos:
            img_url = photo.get("src", {}).get("original")
            if not img_url or img_url in downloaded_urls:
                continue

            try:
                img_data = requests.get(img_url, timeout=15).content
                filename = Path(save_dir) / f"HR{image_counter + 1:03d}.jpg"
                with open(filename, "wb") as f:
                    f.write(img_data)
                downloaded_urls.add(img_url)
                image_counter += 1
                print(f"Downloaded {filename}")
                if image_counter >= total_images:
                    break
            except Exception as e:
                print(f"Failed to download image {img_url}: {e}")

        page += 1
        time.sleep(delay)

    print(f"Downloaded {image_counter} images successfully.")
    return image_counter

# ---------------------------
# Downsample high-res images
# ---------------------------

def create_low_res_images(high_res_dir="data/highRes", low_res_dir="data/lowRes", reduction_factor=10, sharpen_low_res=True):
    high_res_path = Path(high_res_dir)
    low_res_path = Path(low_res_dir)
    low_res_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = sorted([f for f in high_res_path.iterdir() if f.suffix.lower() in valid_exts])

    print(f"Downsampling {len(files)} images by factor {reduction_factor}...")

    for idx, img_file in enumerate(files, start=1):
        try:
            with Image.open(img_file) as img:
                new_w = max(1, img.width // reduction_factor)
                new_h = max(1, img.height // reduction_factor)
                low_res_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                if sharpen_low_res:
                    low_res_img = low_res_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                save_path = low_res_path / f"LR{idx:03d}.jpg"
                low_res_img.save(save_path, "JPEG")
                print(f"Saved {save_path}")
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")

    print("Low-res images created.")

# ---------------------------
# Dataset for paired HR/LR
# ---------------------------

class SRDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, transform=None):
        self.high_res_dir = Path(high_res_dir)
        self.low_res_dir = Path(low_res_dir)
        self.high_res_files = sorted([f for f in self.high_res_dir.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}])
        self.low_res_files = sorted([f for f in self.low_res_dir.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png"}])
        assert len(self.high_res_files) == len(self.low_res_files), "HR and LR datasets size mismatch!"
        self.transform = transform

    def __len__(self):
        return len(self.high_res_files)

    def __getitem__(self, idx):
        hr_img = Image.open(self.high_res_files[idx]).convert("RGB")
        lr_img = Image.open(self.low_res_files[idx]).convert("RGB")
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
        return lr_img, hr_img

# ---------------------------
# Simple SRCNN model example
# ---------------------------

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.layer3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# ---------------------------
# PSNR and SSIM metrics
# ---------------------------

def calculate_metrics(sr, hr):
    # sr and hr expected as numpy arrays with pixel values [0,1]
    psnr = compare_psnr(hr, sr, data_range=1)
    ssim = compare_ssim(hr, sr, multichannel=True, data_range=1)
    return psnr, ssim

# ---------------------------
# Training function
# ---------------------------

def train_sr_model(dataloader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for lr_imgs, hr_imgs in tqdm(dataloader, desc="Training", leave=False):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(lr_imgs)
        loss = criterion(outputs, hr_imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# ---------------------------
# Evaluation function
# ---------------------------

def evaluate_sr_model(dataloader, model, device):
    model.eval()
    psnr_total = 0
    ssim_total = 0
    count = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(dataloader, desc="Evaluating", leave=False):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            outputs = model(lr_imgs).clamp(0, 1)
            # Convert tensors to numpy arrays
            outputs_np = outputs.cpu().permute(0,2,3,1).numpy()
            hr_np = hr_imgs.cpu().permute(0,2,3,1).numpy()
            for sr_img, hr_img in zip(outputs_np, hr_np):
                psnr, ssim = calculate_metrics(sr_img, hr_img)
                psnr_total += psnr
                ssim_total += ssim
                count += 1
    return psnr_total / count, ssim_total / count

# ---------------------------
# Main
# ---------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Download images from Pexels
    download_pexels_images(
        query=args.query,
        total_images=args.total_images,
        per_page=20,
        save_dir=args.high_res_dir,
    )

    # 2. Create low resolution images
    create_low_res_images(
        high_res_dir=args.high_res_dir,
        low_res_dir=args.low_res_dir,
        reduction_factor=args.reduction_factor,
        sharpen_low_res=args.sharpen,
    )

    # 3. Prepare dataset and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to [0,1]
    ])

    dataset = SRDataset(args.high_res_dir, args.low_res_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 4. Initialize model, loss, optimizer
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5. Training loop
    for epoch in range(1, args.epochs + 1):
        loss = train_sr_model(train_loader, model, criterion, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs}, Training Loss: {loss:.6f}")

        # Evaluate every N epochs
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            psnr, ssim = evaluate_sr_model(train_loader, model, device)
            print(f"Evaluation after epoch {epoch}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    model_path = f"checkpoints/srcnn_epoch{args.epochs}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Super-Resolution Training")

    parser.add_argument("--query", type=str, default="tree", help="Search query for images")
    parser.add_argument("--total_images", type=int, default=100, help="Total images to download")
    parser.add_argument("--high_res_dir", type=str, default="data/highRes", help="Directory for high-res images")
    parser.add_argument("--low_res_dir", type=str, default="data/lowRes", help="Directory for low-res images")
    parser.add_argument("--reduction_factor", type=int, default=10, help="Downsampling factor")
    parser.add_argument("--sharpen", type=bool, default=True, help="Sharpen low-res images after downsampling")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N epochs")

    args = parser.parse_args()
    main(args)
