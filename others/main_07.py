import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim


def image_traceability_score(image_data, reference_image=None, weights=None, win_size=0.5):
    """
    Computes a normalized traceability score for an image based on various parameters.

    Args:
        image_data (numpy.ndarray): The input image data.
        reference_image (numpy.ndarray, optional): The reference image data for similarity calculation.
        weights (dict, optional): A dictionary containing weights for each parameter.
            If not provided, equal weights will be assigned.

    Returns:
        float: The normalized traceability score (between 0 and 100).
    """

    if weights is None:
        weights = {
            'sift_features': 0.25,
            'feature_extraction': 0.25,
            'image_quality': 0.25,
            'distribution': 0.25
        }

    # SIFT feature extraction score
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if keypoints is not None:
        sift_features_score = (len(keypoints) / (gray.shape[0] * gray.shape[1])) * 100  # Normalize to image size
    else:
        sift_features_score = 0

        # Use the provided win_size or a smaller value if necessary
    if min(image_data.shape[:2]) < win_size:
        win_size_adjusted = min(image_data.shape[:2]) - 1  # Ensure odd and smaller than image dimensions
        print(f"Warning: Adjusted win_size to {win_size_adjusted} due to small image size.")
    else:
        win_size_adjusted = win_size

    # Other feature extraction score
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    feature_extraction_score = (1 - contrast) * (1 - dissimilarity) * 100  # Normalize between 0-100

    # Image quality score
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    image_quality_score = (blur_score / (blur_score.max() - blur_score.min() + 1e-8)) * 100  # Normalize between 0-100, avoid division by zero

    # Similarity score (optional)
    similarity_score = 0
    if reference_image is not None:
        # Resize images if smaller than 7x7
        if min(image_data.shape[:2]) < 7 or min(reference_image.shape[:2]) < 7:
            resized_image_data = cv2.resize(image_data, (100, 100))
            resized_reference_image = cv2.resize(reference_image, (100, 100))
            similarity_score = ssim(resized_image_data, resized_reference_image, multichannel=True) * 100  # Normalize between 0-100
        else:
            similarity_score = ssim(image_data, reference_image, multichannel=True) * 100  # Normalize between 0-100

    # Distribution score
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    distribution_score = np.sum(histogram * np.log2(histogram + 1e-8)) / (-gray.size) * 100  # Normalize between 0-100

    # Compute the overall traceability score
    overall_score = (weights['sift_features'] * sift_features_score +
                    weights['feature_extraction'] * feature_extraction_score +
                    weights['image_quality'] * image_quality_score +
                    weights['similarity'] * similarity_score +
                    weights['distribution'] * distribution_score)

    return overall_score

if __name__ == '__main__':
    # Replace with your image paths
    image_path = 'test_01.jpeg'
    reference_image_path = 'test_01.jpeg'

    # Load images
    image_data = cv2.imread(image_path)
    reference_image = cv2.imread(reference_image_path)

    if image_data is not None and reference_image is not None:
        # Compute traceability score
        traceability_score = image_traceability_score(image_data, reference_image)
        print(f"Traceability score: {traceability_score:.2f} / 100")
    else:
        print("Failed to load the images.")