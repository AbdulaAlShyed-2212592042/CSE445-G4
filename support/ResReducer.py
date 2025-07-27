import os
from pathlib import Path
from PIL import Image, ImageFilter

def create_low_res_images(
    high_res_dir="data/highRes",
    low_res_dir="data/lowRes",
    reduction_factor=10,
    image_format="JPEG",
    sharpen_low_res=False
):
    """
    Downsamples images from high_res_dir by reduction_factor and saves to low_res_dir.
    
    Args:
        high_res_dir (str): Path to folder with high-res images.
        low_res_dir (str): Path to save low-res images.
        reduction_factor (int): Factor to downscale images by.
        image_format (str): Format to save images.
        sharpen_low_res (bool): If True, apply sharpening filter on downsampled images.
    """
    high_res_path = Path(high_res_dir)
    low_res_path = Path(low_res_dir)
    low_res_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    image_files = [f for f in sorted(high_res_path.iterdir()) if f.suffix.lower() in valid_extensions]

    for idx, img_file in enumerate(image_files, start=1):
        try:
            with Image.open(img_file) as img:
                # Calculate new size
                new_width = max(1, img.width // reduction_factor)
                new_height = max(1, img.height // reduction_factor)

                # Resize with Lanczos filter for quality
                low_res_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Optional sharpening
                if sharpen_low_res:
                    low_res_img = low_res_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

                # Save with new filename
                save_path = low_res_path / f"LR{idx:03d}.jpg"
                low_res_img.save(save_path, image_format)
                print(f"Saved low-res image: {save_path}")

        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")

    print("Image resolution reduction completed!")


if __name__ == "__main__":
    create_low_res_images(
        high_res_dir="data/highRes",
        low_res_dir="data/lowRes",
        reduction_factor=10,
        sharpen_low_res=True  # Set to False if you want no sharpening
    )
