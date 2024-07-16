import cv2
import numpy as np
from skimage import color, filters
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from skimage.metrics import structural_similarity

def extract_dct_features(image):
    dct = cv2.dct(np.float32(image))
    return dct.flatten()[:100]  # Use the first 100 DCT coefficients as features

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
    return lbp_hist / np.sum(lbp_hist)  # Normalize histogram

def extract_sobel_features(image):
    sobel_edges = filters.sobel(image)
    sobel_hist, _ = np.histogram(sobel_edges.ravel(), bins=10)
    return sobel_hist / np.sum(sobel_hist)  # Normalize histogram

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, 1)
    if descriptors is not None:
        return descriptors.flatten()[:100]  # Use the first 100 descriptors
    else:
        return np.zeros(100)  # If no descriptors are found, return zeros

def compute_ssim(image1, image2):
    return structural_similarity(image1, image2, data_range=image1.max() - image1.min())

def quality_assessment(image):
    # Convert to grayscale
    gray_image = color.rgb2gray(image)
    
    # Resize for consistency
    resized_image = resize(gray_image, (256, 256))
    
    # Extract Features
    dct_features = extract_dct_features(resized_image)
    lbp_features = extract_lbp_features(resized_image)
    sobel_features = extract_sobel_features(resized_image)
    sift_features = extract_sift_features(resized_image)
    rotated_image = np.rot90(resized_image, 1)  # Rotate by 90 degrees
    ssim_index = compute_ssim(resized_image, rotated_image)
    
    # Combine features into a single vector
    combined_features = np.concatenate([dct_features, lbp_features, sobel_features, sift_features, [ssim_index]])
    
    # Calculate a pseudo-quality score based on combined features
    quality_score = np.mean(combined_features) * 100 / np.max(combined_features)
    
    return quality_score

def score_image(image_path):
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to read image from {image_path}")
        return 0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Quality assessment
    score = quality_assessment(image)
    
    return score

def main():
    image_path = 'test_01.jpeg'  # Replace with your image path
    score = score_image(image_path)
    if score == 0:
        print("Error: Failed to score the image.")
    else:
        print(f"Image Quality Score: {score:.2f}/100")

if __name__ == "__main__":
    main()