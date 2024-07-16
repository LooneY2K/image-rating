import cv2
import numpy as np

def compute_trackability_score(image_path, max_corners, quality_level, min_distance):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load the image {image_path}")
        return 0

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect corners using FAST
    fast = cv2.FastFeatureDetector_create(threshold=quality_level)
    keypoints = fast.detect(blurred_image, None)
    corners = np.array([kp.pt for kp in keypoints])

    if len(corners) > 0:
        # Refine corner locations
        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_corners = cv2.cornerSubPix(blurred_image, corners, (5, 5), (-1, -1), term_criteria)

        # Feature Density: Normalize the number of detected corners
        feature_density_score = min(len(refined_corners) / max_corners * 100, 100)

        # Feature Distribution: Calculate the distribution score
        height, width = blurred_image.shape
        grid_size = 5
        grid = np.zeros((grid_size, grid_size))
        for corner in refined_corners:
            x, y = int(corner[0] * grid_size / width), int(corner[1] * grid_size / height)
            grid[y, x] += 1
        distribution_score = np.count_nonzero(grid) / (grid_size * grid_size) * 100

        # Feature Distinctiveness: Use FREAK to evaluate the distinctiveness of features
        freak = cv2.FREAK_create()
        _, descriptors = freak.compute(blurred_image, keypoints)
        distinctiveness_score = min(len(descriptors) / max_corners * 100, 100)

        # Combine scores with adjusted weights to align with Vuforia
        trackability_score = (
            feature_density_score * 0.1 +
            distribution_score * 0.6 +
            distinctiveness_score * 0.3
        )
        return trackability_score

    else:
        return 0

# Compute the trackability score for the image
files = ["test_01.jpeg", "test_02.jpeg", "test_03.jpeg", "test_04.jpeg", "test_05.jpeg", "test_06.jpeg"]
for file in files:
    print("processing: ", file)
    trackability_score = compute_trackability_score(file, max_corners=1000, quality_level=25, min_distance=50)
    print(f"Trackability Score: {trackability_score}")
    print("######################################################")