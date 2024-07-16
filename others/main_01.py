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
    Q_LEVEL = 0.01  # Adjusted for more sensitivity
    MIN_DIST = 10

    # Detect corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(equalized_image, maxCorners=MAX_C, qualityLevel=Q_LEVEL, minDistance=MIN_DIST)
    
    if corners is not None:
        corners = np.float32(corners)

        # Corner Density: Normalize the number of detected corners
        corner_density_score = min(len(corners) / MAX_C * 100, 100)

        # Distribution: Calculate the distribution score
        height, width = gray_image.shape
        grid_size = 10
        grid = np.zeros((grid_size, grid_size))
        for corner in corners:
            x, y = int(corner[0][0] * grid_size / width), int(corner[0][1] * grid_size / height)
            grid[y, x] += 1
        distribution_score = np.count_nonzero(grid) / (grid_size * grid_size) * 100

        # Feature Quality: Use Harris corner response to evaluate corner strength
        harris_corners = cv2.cornerHarris(equalized_image, 2, 3, 0.04)
        corner_quality_scores = []
        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            corner_quality_scores.append(harris_corners[y, x])
        quality_score = np.mean(corner_quality_scores)
        quality_score = np.clip(quality_score, 0, 1) * 100

        # Distinctiveness: Use SIFT to evaluate the distinctiveness of features
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(equalized_image, None)
        distinctiveness_score = min(len(keypoints) / MAX_C * 100, 100)

        # Combine scores with adjusted weights to ensure the final score is between 0 and 100
        trackability_score = (
            corner_density_score * 0.4 +
            distribution_score * 0.2 +
            quality_score * 0.2 +
            distinctiveness_score * 0.2
        )
        return trackability_score

    else:
        return 0

# Compute the trackability score for the image
trackability_score = compute_trackability_score("test_01.jpeg")
print(f"Trackability Score: {trackability_score}")



# image_01 - 100, 67
# image_02 - 60, 72
# image_03 - 100, 71
# image_04 - 20, 65
# image_05 - 0 , 67
# image_06 - 100, 68
