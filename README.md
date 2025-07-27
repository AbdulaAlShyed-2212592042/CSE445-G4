# 🌟 Image Super-Resolution Project

---

## 📋 Overview

This project focuses on enhancing the resolution of low-quality images using various **machine learning** and **deep learning** models.  
The primary goal is to generate **high-resolution images** from low-resolution inputs, preserving the original content and structure as accurately as possible.

---

## 👥 Team Members

- **Md. Misbah Khan** (ID: 2132089642)  
- **Md Abdula Al Shyed** (ID: 2212592042)  
- **Rakibul Islam** (ID: 2212058642)  
- **Raju Ahamed Rabby** (ID: 2212592042)  

---

## 🧠 Models Implemented

1. **XGBoost**  
   An open-source machine learning library known for its speed and efficiency in supervised learning tasks.

2. **Random Forest**  
   An ensemble learning method that constructs multiple decision trees to improve predictive performance.

3. **Convolutional Neural Networks (CNNs)**  
   Deep learning models designed for processing structured grid data, such as images, by utilizing convolutional layers to automatically learn spatial hierarchies of features.

4. **Super-Resolution Generative Adversarial Networks (SRGANs)**  
   Deep learning models that enhance image resolution by generating high-quality images from low-resolution inputs.

---

## 📂 Dataset Preparation

- **Source:**  
  Downloaded 100 high-resolution images from [Pexels](https://www.pexels.com).

- **Processing:**  
  - Resized images using the **Pillow** library.  
  - Saved high-resolution images in `data/highRes` folder with sequential filenames:  
    `HR001.jpg`, `HR002.jpg`, ..., `HR100.jpg`.

- **Downsampling:**  
  - Created low-resolution images by downsampling with a factor of **10** using the **Lanczos** resampling filter.  
  - Saved in `data/lowRes` folder with filenames:  
    `LR001.jpg`, `LR002.jpg`, ..., `LR100.jpg`.

---

## ❓ Problem Statement

Upscaling low-resolution images to high-resolution ones is challenging. Traditional methods like **K-Nearest Neighbors (KNN)** and **Bilinear Interpolation** often produce images with larger pixels rather than truly enhancing details.  

According to the **Data Processing Inequality**, processing cannot add new information, implying the need for advanced models to infer missing details intelligently.  

Hence, deep learning models such as **CNNs** and **SRGANs** are employed to learn complex mappings and reconstruct high-resolution images from low-resolution inputs.

---

## 🛠 Approach

- Implemented and evaluated four models: XGBoost, Random Forest, CNN, and SRGAN.
- Used the same dataset for training and testing to ensure fair comparisons.
- Evaluated image quality using metrics:  
  - **Peak Signal-to-Noise Ratio (PSNR)**  
  - **Structural Similarity Index Measure (SSIM)**

---

## 📈 Results

- Model performances were compared using PSNR and SSIM scores.  
- Detailed results and comparative analyses are available in the `results` folder.

---

## ⚙️ Requirements

- **Python**: Version 3.11  
- **Libraries**:  
  - `numpy`  
  - `pandas`  
  - `scikit-learn`  
  - `tensorflow`  
  - `Pillow`

---

## 🚀 Installation Steps

1. **Clone the Repository**  
   Open your terminal or command prompt and run:  
   ```bash
   git clone https://github.com/MisbahKhan0009/CSE445-G4.git
# 🌟 Image Super-Resolution Project

---

## 📋 Overview

This project focuses on enhancing the resolution of low-quality images using various **machine learning** and **deep learning** models.  
The primary goal is to generate **high-resolution images** from low-resolution inputs, preserving the original content and structure as accurately as possible.

---

## 👥 Team Members

- **Md. Misbah Khan** (ID: 2132089642)  
- **Md Abdula Al Shyed** (ID: 2212592042)  
- **Rakibul Islam** (ID: 2212058642)  
- **Raju Ahamed Rabby** (ID: 2212592042)  

---

## 🧠 Models Implemented

1. **XGBoost**  
   An open-source machine learning library known for its speed and efficiency in supervised learning tasks.

2. **Random Forest**  
   An ensemble learning method that constructs multiple decision trees to improve predictive performance.

3. **Convolutional Neural Networks (CNNs)**  
   Deep learning models designed for processing structured grid data, such as images, by utilizing convolutional layers to automatically learn spatial hierarchies of features.

4. **Super-Resolution Generative Adversarial Networks (SRGANs)**  
   Deep learning models that enhance image resolution by generating high-quality images from low-resolution inputs.

---

## 📂 Dataset Preparation

- **Source:**  
  Downloaded 100 high-resolution images from [Pexels](https://www.pexels.com).

- **Processing:**  
  - Resized images using the **Pillow** library.  
  - Saved high-resolution images in `data/highRes` folder with sequential filenames:  
    `HR001.jpg`, `HR002.jpg`, ..., `HR100.jpg`.

- **Downsampling:**  
  - Created low-resolution images by downsampling with a factor of **10** using the **Lanczos** resampling filter.  
  - Saved in `data/lowRes` folder with filenames:  
    `LR001.jpg`, `LR002.jpg`, ..., `LR100.jpg`.

---

## ❓ Problem Statement

Upscaling low-resolution images to high-resolution ones is challenging. Traditional methods like **K-Nearest Neighbors (KNN)** and **Bilinear Interpolation** often produce images with larger pixels rather than truly enhancing details.  

According to the **Data Processing Inequality**, processing cannot add new information, implying the need for advanced models to infer missing details intelligently.  

Hence, deep learning models such as **CNNs** and **SRGANs** are employed to learn complex mappings and reconstruct high-resolution images from low-resolution inputs.

---

## 🛠 Approach

- Implemented and evaluated four models: XGBoost, Random Forest, CNN, and SRGAN.
- Used the same dataset for training and testing to ensure fair comparisons.
- Evaluated image quality using metrics:  
  - **Peak Signal-to-Noise Ratio (PSNR)**  
  - **Structural Similarity Index Measure (SSIM)**

---

## 📈 Results

- Model performances were compared using PSNR and SSIM scores.  
- Detailed results and comparative analyses are available in the `results` folder.

---

## ⚙️ Requirements

- **Python**: Version 3.11  
- **Libraries**:  
  - `numpy`  
  - `pandas`  
  - `scikit-learn`  
  - `tensorflow`  
  - `Pillow`

---

## 🚀 Installation Steps

1. **Clone the Repository**  
   Open your terminal or command prompt and run:  
   ```bash
   git clone https://github.com/MisbahKhan0009/CSE445-G4.git
