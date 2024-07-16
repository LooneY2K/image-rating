import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def detect_and_compute(image, detector):
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors


def calculate_uniqueness_score(descriptors):
    if len(descriptors) == 0:
        return 0
    distances = euclidean_distances(descriptors)
    np.fill_diagonal(distances, np.inf)  # Ignore self-distances
    min_distances = np.min(distances, axis=1)
    uniqueness_score = np.mean(min_distances)
    return uniqueness_score


def calculate_coverage_score(keypoints, image_shape):
    if len(keypoints) == 0:
        return 0
    height, width = image_shape[:2]
    points = np.array([kp.pt for kp in keypoints])
    x_coverage = np.ptp(points[:, 0]) / width
    y_coverage = np.ptp(points[:, 1]) / height
    coverage_score = (x_coverage + y_coverage) / 2
    return coverage_score


# Initialize feature detector and descriptor (e.g., SIFT)
detector = cv2.SIFT_create()


# Calculate scores
def calculate_feature_score(image):
    keypoints, descriptors = detect_and_compute(image, detector)
    if len(keypoints) == 0 or descriptors is None:
        return 0  # Return a score of 0 if no keypoints or descriptors are found

    uniqueness = calculate_uniqueness_score(descriptors)
    coverage = calculate_coverage_score(keypoints, image.shape)
    num_features = len(keypoints)

    # Normalize scores
    normalized_uniqueness = min(uniqueness / 1000, 1.0)  # Adjust normalization factor
    normalized_coverage = min(coverage, 1.0)  # Coverage is naturally between 0 and 1
    normalized_num_features = min(
        num_features / 1000, 1.0
    )  # Adjust normalization factor

    # Combine scores with weights
    weights = {"uniqueness": 0.4, "coverage": 0.2, "num_features": 0.4}

    combined_score = (
        weights["uniqueness"] * normalized_uniqueness
        + weights["coverage"] * normalized_coverage
        + weights["num_features"] * normalized_num_features
    ) * 100  # Scale to 0-100

    return combined_score



# # Load and process images
# files = [
#     "new_images/test_01.png",
#     "dataset/test_01.png",
#     "dataset/test_02.png",
#     "dataset/test_03.png",
#     "dataset/test_04.png",
#     "dataset/test_05.png",
#     "dataset/test_09.png",
#     "dataset/test_10.png",
#     "dataset/test_11.png",
#     "dataset/test_12.png",
#     "dataset/test_13.png",
#     "dataset/test_14.png",
#     "dataset/test_15.png",
#     "dataset/test_16.png",
#     "dataset/test_17.png",
#     "dataset/test_18.png",
#     "dataset/test_19.png",
#     "dataset/test_20.png",
#     "dataset/test_21.png",
#     "dataset/test_22.png",
#     "dataset/test_23.png",
#     "dataset/test_24.png",
#     "dataset/test_25.png",
#     "dataset/test_26.png",
#     "dataset/test_27.png",
#     "dataset/test_29.png",
#     "dataset/test_30.png",
#     "dataset/test_31.png",
#     "dataset/test_32.png",
#     "dataset/test_33.png",
#     "dataset/test_34.png",
#     "dataset/test_35.png",
#     "dataset/test_36.png",
#     "dataset/test_37.png",
#     "dataset/test_38.png",
#     "dataset/test_39.png",
#     "dataset/test_40.png",
#     "dataset/test_41.png",
#     "dataset/test_42.png",
#     "dataset/test_43.png",
#     "dataset/test_44.png",
#     "dataset/test_45.png",
#     "dataset/test_46.png",
#     "dataset/test_47.png",
#     "dataset/test_49.png",
#     "dataset/test_50.png",
#     "dataset/test_51.png",
#     "dataset/test_52.png",
#     "dataset/test_53.png",
#     "dataset/test_54.png",
#     "dataset/test_55.png",
#     "dataset/test_56.png",
#     "dataset/test_57.png",
#     "dataset/test_58.png",
# ]

# for file in files:
#     print("Processing:", file)
#     image = cv2.imread(file)
#     if image is not None:
#         score = calculate_feature_score(image)
#         print(f"Image Feature Score: {score:.2f}")
#     else:
#         print(f"Error: Unable to read {file}")
