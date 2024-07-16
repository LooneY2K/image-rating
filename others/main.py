import cv2
import numpy as np

def compute_trackability_score(image_path):
    # Load and preprocess the image
    image1 = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    equalized_image = cv2.equalizeHist(blurred_image)

    # Parameters
    MAX_C = 300
    Q_LEVEL = 0.3
    MIN_DIST = 10
    FACTOR = MAX_C / 100

    # Detect corners
    corners1 = cv2.goodFeaturesToTrack(equalized_image, maxCorners=MAX_C, qualityLevel=Q_LEVEL, minDistance=MIN_DIST, useHarrisDetector=True, k=0.04)
    if corners1 is not None:
        corners1 = np.float32(corners1)

        # Calculate robust feature descriptors (SIFT)
        sift = cv2.SIFT_create()
        keypoints = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in corners1]
        keypoints, descriptors = sift.compute(gray_image, keypoints)

        # Corner Density: Normalize the number of detected corners
        corner_density_score = len(corners1) / MAX_C * 100

        # Distribution: Calculate the distribution score
        height, width = gray_image.shape
        distribution_score = 0
        if len(corners1) > 0:
            grid_size = 10
            grid = np.zeros((grid_size, grid_size))
            for corner in corners1:
                x, y = int(corner[0][0] * grid_size / width), int(corner[0][1] * grid_size / height)
                grid[y, x] += 1
            distribution_score = np.count_nonzero(grid) / (grid_size * grid_size) * 100

        # Feature Quality: Calculate quality score based on the strength of the corners
        quality_score = np.mean([cv2.cornerHarris(gray_image, 2, 3, 0.04)[int(corner[0][1]), int(corner[0][0])] for corner in corners1])
        quality_score = min(max(quality_score * 100, 0), 100)

        # Combine scores
        trackability_score = (corner_density_score * 0.4) + (distribution_score * 0.3) + (quality_score * 0.3)
        return trackability_score

    else:
        return 0

# Compute the trackability score for the image
trackability_score = compute_trackability_score("test_02.jpeg")
print(f"Trackability Score: {trackability_score}")

# MAX_C = 300
# Q_LEVEL = 0.3
# MIN_DIST = 10
# FACTOR = MAX_C / 100

# image_01 - 100
# image_02 - 60
# image_03 - 100
# image_04 - 20
# image_05 - 0
# image_06 - 100