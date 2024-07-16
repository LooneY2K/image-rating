import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy


# Function to detect keypoints and compute descriptors
def detect_and_compute(image, detector):
    keypoints = detector.detect(image, None)
    keypoints, descriptors = detector.compute(image, keypoints)
    return keypoints, descriptors


# Function to calculate spatial distribution score
def calculate_spatial_distribution(keypoints, image_size):
    height, width = image_size
    x_coords = [kp.pt[0] for kp in keypoints]
    y_coords = [kp.pt[1] for kp in keypoints]
    x_hist, _ = np.histogram(x_coords, bins=10, range=(0, width))
    y_hist, _ = np.histogram(y_coords, bins=10, range=(0, height))
    spatial_distribution_score = entropy(x_hist) + entropy(y_hist)
    return spatial_distribution_score


# Function to calculate descriptor uniqueness score
def calculate_uniqueness_score(descriptors):
    distances = pairwise_distances(descriptors, metric="euclidean")
    min_distances = np.min(distances + np.eye(len(distances)) * distances.max(), axis=1)
    uniqueness_score = np.mean(min_distances)
    return uniqueness_score


# Function to calculate coverage score
def calculate_coverage(keypoints, image_size):
    height, width = image_size
    grid_size = 10
    grid = np.zeros((grid_size, grid_size))
    for kp in keypoints:
        x, y = kp.pt
        grid[int(y * grid_size / height), int(x * grid_size / width)] += 1
    coverage_score = np.count_nonzero(grid) / (grid_size * grid_size)
    return coverage_score


# Function to calculate entropy of keypoints
def calculate_keypoint_entropy(keypoints, image_size):
    height, width = image_size
    x_coords = [kp.pt[0] for kp in keypoints]
    y_coords = [kp.pt[1] for kp in keypoints]
    positions = np.vstack((x_coords, y_coords)).T
    hist, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=10, range=[[0, width], [0, height]]
    )
    keypoint_entropy = entropy(hist.flatten())
    return keypoint_entropy


# Function to calculate grid-based uniformity score
def calculate_grid_uniformity(keypoints, image_size):
    height, width = image_size
    grid_size = 10
    grid = np.zeros((grid_size, grid_size))
    for kp in keypoints:
        x, y = kp.pt
        grid[int(y * grid_size / height), int(x * grid_size / width)] += 1
    grid_mean = np.mean(grid)
    grid_std = np.std(grid)
    uniformity_score = 1 - (grid_std / grid_mean)
    return uniformity_score


# Function to calculate the overall distribution score
def calculate_distribution_score(image):
    # Detect and compute descriptors
    detector = cv2.SIFT_create()
    keypoints, descriptors = detect_and_compute(image, detector)

    # Image size
    image_size = image.shape[:2]

    # Calculate individual distribution metrics
    spatial_distribution = calculate_spatial_distribution(keypoints, image_size)
    uniqueness = calculate_uniqueness_score(descriptors)
    coverage = calculate_coverage(keypoints, image_size)
    keypoint_entropy = calculate_keypoint_entropy(keypoints, image_size)
    grid_uniformity = calculate_grid_uniformity(keypoints, image_size)

    # Normalize scores (example normalization factors)
    spatial_distribution_normalized = min(
        spatial_distribution / 10.0, 1.0
    )  # Example normalization factor
    uniqueness_normalized = min(uniqueness / 100.0, 1.0)  # Example normalization factor
    coverage_normalized = coverage  # Coverage is already between 0 and 1
    keypoint_entropy_normalized = min(
        keypoint_entropy / 10.0, 1.0
    )  # Example normalization factor
    grid_uniformity_normalized = (
        grid_uniformity  # Uniformity is already between 0 and 1
    )

    # Print
    # print(f"Spatial distribution score is : {spatial_distribution}")
    # print(f"Uniqueness normalized score is : {uniqueness_normalized}")
    # print(f"Coverage Normalized score is : {coverage_normalized}")
    # print(f"Keypoint entropy score is : {keypoint_entropy_normalized}")
    # print(f"Grid uniformity score is: {grid_uniformity_normalized}")
    # Combine scores with weights
    weights = {
        "spatial_distribution": 0.2,
        "uniqueness": 0.2,
        "coverage": 0.2,
        "keypoint_entropy": 0.2,
        "grid_uniformity": 0.2,
    }

    combined_score = (
        weights["spatial_distribution"] * spatial_distribution_normalized
        + weights["uniqueness"] * uniqueness_normalized
        + weights["coverage"] * coverage_normalized
        + weights["keypoint_entropy"] * keypoint_entropy_normalized
        + weights["grid_uniformity"] * grid_uniformity_normalized
    )

    # print(f"spatial distribution: {spatial_distribution_normalized}")
    # print(f"uniqueness: {uniqueness_normalized}")
    # print(f"coverage distribution: {coverage_normalized}")
    # print(f"grid uniformity: {grid_uniformity_normalized}")
    # print(f"keypoint entropy: {keypoint_entropy_normalized}")

    final_distribution_score = combined_score * 100  # Scale to 0-100

    return final_distribution_score


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
# ]
# for file in files:
#     print("Processing:", file)
#     image = cv2.imread(file)
#     if image is not None:
#         distribution_score = calculate_distribution_score(image)
#         print(f"Image Distribution Score: {distribution_score:.2f}")
#     else:
#         print(f"Error: Unable to read {file}")
