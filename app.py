import cv2
import numpy as np
import csv
from distinctiveness import *
from distribution import *
from quality_score import *
from feature_score import *
import os


def detect_and_compute(image, detector):
    if image is None:
        raise ValueError("Image is empty or not found.")
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors


def draw_and_save_keypoints(image, keypoints, output_path):
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )
    # Save the image to the specified output path
    cv2.imwrite(output_path, image_with_keypoints)
    print(f"Image saved to {output_path}")


files = [
    "new_images/test_01.png",
    "dataset/test_01.png",
    "dataset/test_02.png",
    "dataset/test_03.png",
    "dataset/test_04.png",
    "dataset/test_05.png",
    "dataset/test_09.png",
    "dataset/test_10.png",
    "dataset/test_11.png",
    "dataset/test_12.png",
    "dataset/test_13.png",
    "dataset/test_14.png",
    "dataset/test_15.png",
    "dataset/test_16.png",
    "dataset/test_17.png",
    "dataset/test_18.png",
    "dataset/test_19.png",
    "dataset/test_20.png",
    "dataset/test_21.png",
    "dataset/test_22.png",
    "dataset/test_23.png",
    "dataset/test_24.png",
    "dataset/test_25.png",
    "dataset/test_26.png",
    "dataset/test_27.png",
    "dataset/test_29.png",
    "dataset/test_30.png",
    "dataset/test_31.png",
    "dataset/test_32.png",
    "dataset/test_33.png",
    "dataset/test_34.png",
    "dataset/test_35.png",
    "dataset/test_36.png",
    "dataset/test_37.png",
    "dataset/test_38.png",
    "dataset/test_39.png",
    "dataset/test_40.png",
    "dataset/test_41.png",
    "dataset/test_42.png",
    "dataset/test_43.png",
    "dataset/test_44.png",
    "dataset/test_45.png",
    "dataset/test_46.png",
    "dataset/test_47.png",
    "dataset/test_49.png",
    "dataset/test_50.png",
    "dataset/test_51.png",
    "dataset/test_52.png",
    "dataset/test_53.png",
    "dataset/test_54.png",
    "dataset/test_55.png",
    "dataset/test_56.png",
    "dataset/test_57.png",
    "dataset/test_58.png",
    "dataset/vida.jpg",
    "dataset/nothing.png",
]

# Ensure the "show" directory exists
os.makedirs("show_single", exist_ok=True)

with open("final_image_scores.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "Image",
            "Feature Score",
            "Quality Score",
            "Distribution Score",
            "Distinctiveness Score",
            "Total Score",
            "New Score",
            "Good or Bad",
            "Stars",
        ]
    )

    for file in files:
        print("Processing:", file)
        image = cv2.imread(file)
        if image is None:
            print(f"Error: Unable to read {file}")
            writer.writerow(
                [
                    file,
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                ]
            )
            continue

        try:
            detector = cv2.SIFT_create()
            keypoints, _ = detect_and_compute(image, detector)

            quality_score = calculate_quality_score(image)
            feature_score = calculate_feature_score(image)
            distinct_score = calculate_distinctiveness_score(image, detector)
            distribution_score = calculate_distribution_score(image)

            if quality_score < 35:
                weights = {
                    "distinct": 0.25,
                    "distribution": 0.35,
                    "quality": 0.2,
                    "feature": 0.2,
                }
            elif distribution_score < 30:
                weights = {
                    "distinct": 0.25,
                    "distribution": 0.15,
                    "quality": 0.4,
                    "feature": 0.2,
                }
            else:
                weights = {
                    "distinct": 0.4,
                    "distribution": 0.1,
                    "quality": 0.1,
                    "feature": 0.4,
                }

            total_score = (
                weights["distinct"] * distinct_score
                + weights["distribution"] * distribution_score
                + weights["quality"] * quality_score
                + weights["feature"] * feature_score
            )
            new_score = total_score / 20

            if new_score >= 2.8 and new_score < 3.5:
                affr = "Good"
                stars = min(round(new_score), 5)
            elif new_score >= 3.5 and new_score < 3.7:
                affr = "Good"
                stars = min(round(new_score), 5)
            elif new_score >= 3.7:
                affr = "Good"
                stars = min(round(new_score), 5) + 1  # Cap stars at 5
            else:
                affr = "Bad"
                stars = max(round(new_score) - 1, 0)  # Ensure stars is non-negative

            writer.writerow(
                [
                    file,
                    f"{feature_score:.2f}",
                    f"{quality_score:.2f}",
                    f"{distribution_score:.2f}",
                    f"{distinct_score:.2f}",
                    f"{total_score:.2f}",
                    f"{new_score:.2f}",
                    affr,
                    stars,
                ]
            )

            print(f"Image Feature Score: {feature_score:.2f}")
            print(f"Quality score: {quality_score:.2f}")
            print(f"Distinctiveness score: {distinct_score:.2f}")
            print(f"Distribution score: {distribution_score:.2f}")
            print(f"The overall total score: {total_score:.2f}")
            print(f"New score: {new_score:.2f}")
            print(f"{file} is a {affr.lower()} image with {stars} stars")

            # Save image with keypoints
            output_path = f"./show_single/{os.path.basename(file)}"
            draw_and_save_keypoints(image, keypoints, output_path)

        except Exception as e:
            print(f"Error processing {file}: {e}")
            writer.writerow(
                [
                    file,
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                ]
            )

print("CSV file 'final_image_scores.csv' has been created with all scores.")
