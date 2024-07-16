import cv2
import numpy as np

def compute_trackability_score(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    equalized_image = cv2.equalizeHist(blurred_image)

    # Parameters
    MAX_CORNERS = 200  # Max corners to detect
    QUALITY_LEVEL = 0.01  # Quality level for corner detection
    MIN_DISTANCE = 20 # Minimum distance between corners

    # Detect corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(equalized_image, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE)
    
    if corners is not None:
        corners = np.float32(corners)

        # Feature Density: Normalize the number of detected corners
        num_corners = len(corners)
        feature_density_score = min(num_corners / MAX_CORNERS * 100, 100)

        # Feature Distribution: Calculate the distribution score
        height, width = gray_image.shape
        grid_size = 10
        grid = np.zeros((grid_size, grid_size))
        for corner in corners:
            x, y = int(corner[0][0] * grid_size / width), int(corner[0][1] * grid_size / height)
            grid[y, x] += 1
        distribution_score = np.count_nonzero(grid) / (grid_size * grid_size) * 100
        distribution_score = min(distribution_score, 100)

        # Feature Quality: Use both Harris corner response and minimum eigenvalue to evaluate corner strength
        harris_corners = cv2.cornerHarris(equalized_image, 2, 3, 0.04)
        min_eigenval = cv2.cornerMinEigenVal(equalized_image, 3)
        corner_quality_scores = []
        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            corner_quality_scores.append((harris_corners[y, x] + min_eigenval[y, x]) / 2)
        quality_score = np.mean(corner_quality_scores)
        quality_score = min(max(quality_score, 0), 1) * 100

        # Feature Distinctiveness: Use ORB descriptors to evaluate the distinctiveness of features
        orb = cv2.AKAZE_create()
        keypoints, descriptors = orb.detectAndCompute(equalized_image, None)
        distinctiveness_score = min(len(keypoints) / MAX_CORNERS * 100, 100)

        # Combine scores with adjusted weights to ensure the final score is between 0 and 100
        print("feature_density_score: ", feature_density_score)
        print("distribution_score: ", distribution_score)
        print("quality_score: ", quality_score)
        print("distinctiveness_score: ", distinctiveness_score)
        trackability_score = (
            feature_density_score * 0.25 +
            distribution_score * 0.25 +
            quality_score * 0.25 +
            distinctiveness_score * 0.25
        )
        return trackability_score

    else:
        return 0
# Compute the trackability score for the image

files = ["test_01.jpeg", "test_02.jpeg", "test_03.jpeg", "test_04.jpeg", "test_05.jpeg", "test_06.jpeg"]
for file in files:
    print("processing: ", file)
    trackability_score = compute_trackability_score(file)
    print(f"Trackability Score: {trackability_score}")


# image_01 - 100
# image_02 - 60
# image_03 - 100
# image_04 - 20
# image_05 - 0
# image_06 - 100


