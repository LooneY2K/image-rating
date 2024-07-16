import cv2
import numpy as np
from skimage import color, filters


# Function to calculate sharpness using the variance of the Laplacian
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


# Function to calculate global contrast
def calculate_global_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast


# Function to calculate colorfulness
def calculate_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    colorfulness = np.sqrt((std_rg**2) + (std_yb**2)) + (
        0.3 * np.sqrt((mean_rg**2) + (mean_yb**2))
    )
    return colorfulness


# Function to calculate noise level using wavelet decomposition
def calculate_noise_level(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_level = filters.threshold_yen(gray)
    return noise_level


# Function to calculate the overall quality score
def calculate_quality_score(image):
    # Calculate individual quality metrics
    sharpness = calculate_sharpness(image)
    contrast = calculate_global_contrast(image)
    colorfulness = calculate_colorfulness(image)
    noise_level = calculate_noise_level(image)

    # Normalize scores (example normalization factors)
    sharpness_normalized = min(
        sharpness / 1000.0, 1.0
    )  # Assume max sharpness is 1000 for normalization
    contrast_normalized = min(
        contrast / 128.0, 1.0
    )  # Assume max contrast (std dev) is 128
    colorfulness_normalized = min(
        colorfulness / 100.0, 1.0
    )  # Assume max colorfulness is 100
    noise_level_normalized = 1.0 - min(
        noise_level / 255.0, 1.0
    )  # Invert and normalize noise level

    # Combine scores with weights
    weights = {
        "sharpness": 0.25,
        "contrast": 0.25,
        "colorfulness": 0.25,
        "noise_level": 0.25,
    }

    combined_score = (
        weights["sharpness"] * sharpness_normalized
        + weights["contrast"] * contrast_normalized
        + weights["colorfulness"] * colorfulness_normalized
        + weights["noise_level"] * noise_level_normalized
    )

    # print(f"The sharpness is: {sharpness_normalized}")
    # print(f"Contrast is: {contrast_normalized}")
    # print(f"Colorfulness is: {colorfulness_normalized}")
    # print(f"Noise level is: {noise_level_normalized}")
    final_quality_score = combined_score * 100  # Scale to 0-100

    return final_quality_score


# # Load image
# files = [
#     "Images/test_01.jpeg",
#     "Images/test_02.jpeg",
#     "Images/test_03.jpeg",
#     "Images/test_04.jpeg",
#     "Images/test_05.jpeg",
#     "Images/test_06.jpeg",
#     "Images/test_07.jpg",
#     "Images/test_08.jpg",
#     "Images/test_09.png",
#     "Images/test_10.jpg",
#     "Images/test_11.png",
#     "Images/test_12.png",
#     "Images/test_13.png",
#     "Images/test_14.jpg",
#     "Images/test_15.png",
#     "Images/test_16.jpeg",
#     "Images/test_17.jpg",
# ]
# for file in files:
#     print("Processing:", file)
#     image = cv2.imread(file)
#     if image is not None:
#         quality_score = calculate_quality_score(image)
#         print(f"Image Quality Score: {quality_score:.2f}")
#     else:
#         print(f"Error: Unable to read {file}")
