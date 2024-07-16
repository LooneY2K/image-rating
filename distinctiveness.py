import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy


def detect_and_compute(image, detector):
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors


def calculate_uniqueness_score(descriptors):
    if descriptors is None or len(descriptors) == 0:
        return 0
    distances = pairwise_distances(descriptors, metric="euclidean")
    min_distances = np.min(distances + np.eye(len(distances)) * distances.max(), axis=1)
    uniqueness_score = np.mean(min_distances)
    return uniqueness_score


def calculate_keypoint_strength(keypoints):
    if not keypoints:
        return 0
    strength_score = np.mean([kp.response for kp in keypoints])
    return strength_score


def calculate_repeatability_score(keypoints, transformed_keypoints):
    repeatability_score = len(
        set(keypoints).intersection(set(transformed_keypoints))
    ) / len(keypoints)
    return repeatability_score


def calculate_descriptor_diversity(descriptors):
    if descriptors is None or len(descriptors) == 0:
        return 0
    descriptor_variance = np.var(descriptors, axis=0)
    diversity_score = np.mean(descriptor_variance)
    return diversity_score


def calculate_matching_accuracy(keypoints, matched_keypoints):
    matching_accuracy_score = len(
        set(keypoints).intersection(set(matched_keypoints))
    ) / len(keypoints)
    return matching_accuracy_score


def calculate_spatial_distribution(keypoints, image_size):
    if not keypoints:
        return 0
    height, width = image_size
    x_coords = [kp.pt[0] for kp in keypoints]
    y_coords = [kp.pt[1] for kp in keypoints]
    x_hist, _ = np.histogram(x_coords, bins=10, range=(0, width))
    y_hist, _ = np.histogram(y_coords, bins=10, range=(0, height))
    spatial_distribution_score = entropy(x_hist) + entropy(y_hist)
    return spatial_distribution_score


def calculate_distinctiveness_score(
    image, detector, transformed_image=None, matched_keypoints=None
):
    keypoints, descriptors = detect_and_compute(image, detector)

    if len(keypoints) == 0 or descriptors is None:
        return 0

    uniqueness = calculate_uniqueness_score(descriptors)
    strength = calculate_keypoint_strength(keypoints)
    diversity = calculate_descriptor_diversity(descriptors)
    spatial_distribution = calculate_spatial_distribution(keypoints, image.shape[:2])

    repeatability = (
        1.0
        if transformed_image is None
        else calculate_repeatability_score(
            keypoints, detect_and_compute(transformed_image, detector)[0]
        )
    )
    matching_accuracy = (
        1.0
        if matched_keypoints is None
        else calculate_matching_accuracy(keypoints, matched_keypoints)
    )

    uniqueness_normalized = min(uniqueness / 10.0, 1.0) * 100
    strength_normalized = min(strength / 0.1, 1.0) * 100
    diversity_normalized = min(diversity / 0.01, 1.0) * 100
    spatial_distribution_normalized = min(spatial_distribution / 2.0, 1.0) * 100
    repeatability_normalized = repeatability * 100
    matching_accuracy_normalized = matching_accuracy * 100

    weights = {
        "uniqueness": 0.3,
        "strength": 0.2,
        "diversity": 0.3,
        "spatial_distribution": 0.1,
        "repeatability": 0.05,
        "matching_accuracy": 0.05,
    }

    combined_score = (
        weights["uniqueness"] * uniqueness_normalized
        + weights["strength"] * strength_normalized
        + weights["diversity"] * diversity_normalized
        + weights["spatial_distribution"] * spatial_distribution_normalized
        + weights["repeatability"] * repeatability_normalized
        + weights["matching_accuracy"] * matching_accuracy_normalized
    )

    final_distinctiveness_score = combined_score  # Already scaled to 0-100

    return final_distinctiveness_score
