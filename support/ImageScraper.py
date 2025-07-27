import requests
import os
from dotenv import load_dotenv
import time

def download_pexels_images(
    query: str,
    total_images: int,
    per_page: int = 20,
    save_dir: str = "data/highRes",
    api_key_env: str = "PEXELS_API_KEY",
    max_retries: int = 3,
    delay_between_requests: float = 1.0,
):
    """
    Download images from Pexels based on a search query.

    Args:
        query (str): Search keyword for images.
        total_images (int): Total number of unique images to download.
        per_page (int): Number of images per API request page (max 80).
        save_dir (str): Directory to save downloaded images.
        api_key_env (str): Environment variable name storing the Pexels API key.
        max_retries (int): Number of retries for failed API requests.
        delay_between_requests (float): Delay between API requests in seconds to avoid rate limits.

    Returns:
        int: Number of images downloaded successfully.
    """
    load_dotenv()
    API_KEY = os.getenv(api_key_env)
    if not API_KEY:
        raise ValueError(f"API key not found in environment variable '{api_key_env}'")

    os.makedirs(save_dir, exist_ok=True)

    URL = "https://api.pexels.com/v1/search"
    headers = {"Authorization": API_KEY}

    image_counter = 0
    page = 1
    downloaded_urls = set()

    while image_counter < total_images:
        params = {"query": query, "per_page": per_page, "page": page}
        for attempt in range(max_retries):
            try:
                response = requests.get(URL, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    break  # Success
                else:
                    print(f"Error fetching images (Status {response.status_code}): {response.text}")
                    time.sleep(delay_between_requests)
            except requests.RequestException as e:
                print(f"Request failed: {e}. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(delay_between_requests)
        else:
            print("Max retries reached. Stopping download.")
            break

        data = response.json()
        photos = data.get("photos", [])
        if not photos:
            print("No more photos found. Ending download.")
            break

        for photo in photos:
            image_url = photo.get("src", {}).get("original")
            if not image_url:
                continue
            if image_url in downloaded_urls:
                continue

            try:
                image_response = requests.get(image_url, timeout=15)
                if image_response.status_code == 200:
                    filename = os.path.join(save_dir, f"HR{image_counter + 1:03d}.jpg")
                    with open(filename, "wb") as f:
                        f.write(image_response.content)
                    downloaded_urls.add(image_url)
                    image_counter += 1
                    print(f"Downloaded image {image_counter}: {filename}")
                else:
                    print(f"Failed to download image {image_url} (Status {image_response.status_code})")
            except requests.RequestException as e:
                print(f"Error downloading image {image_url}: {e}")

            if image_counter >= total_images:
                break

        page += 1
        time.sleep(delay_between_requests)

    print(f"\nDownloaded {image_counter} unique images successfully!")
    return image_counter


if __name__ == "__main__":
    # Example usage
    download_pexels_images(
        query="tree",
        total_images=100,
        per_page=20,
        save_dir="data/highRes"
    )
