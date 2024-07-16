import cv2
import numpy as np

def compute_trackability_score(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Parameters
    MAX_CORNERS = 200  # Max corners to detect
    QUALITY_LEVEL = 0.01  # Quality level for corner detection
    MIN_DISTANCE = 20  # Minimum distance between corners

    # Detect corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(blurred_image, maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE)
    
    if corners is not None:
        corners = np.float32(corners)

        # Feature Density: Normalize the number of detected corners
        feature_density_score = min(len(corners) / MAX_CORNERS * 100, 100)

        # Feature Distribution: Calculate the distribution score
        height, width = blurred_image.shape
        grid_size = 10
        grid = np.zeros((grid_size, grid_size))
        for corner in corners:
            x, y = int(corner[0][0] * grid_size / width), int(corner[0][1] * grid_size / height)
            grid[y, x] += 1
        distribution_score = np.count_nonzero(grid) / (grid_size * grid_size) * 100

        # Feature Quality: Use Minimum Eigenvalue to evaluate corner strength
        min_eigenval = cv2.cornerMinEigenVal(blurred_image, 3)
        quality_score = np.mean(min_eigenval)
        quality_score = min(max(quality_score, 0), 1) * 100

        # Feature Distinctiveness: Use ORB to evaluate the distinctiveness of features
        orb = cv2.SIFT_create(nfeatures=1000)
        keypoints = orb.detect(blurred_image, None)
        distinctiveness_score = min(len(keypoints) / MAX_CORNERS * 100, 100)

        # Combine scores with adjusted weights to ensure the final score is between 0 and 100
        print("feature_density_score: ", feature_density_score)
        print("distribution_score: ", distribution_score)
        print("quality_score: ", quality_score)
        print("distinctiveness_score: ", distinctiveness_score)
        trackability_score = (
            feature_density_score * 0.05 +
            distribution_score * 0.85 +
            quality_score * 0.05 +
            distinctiveness_score * 0.05
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
    print("######################################################")



# image_01 - 100
# image_02 - 60
# image_03 - 100
# image_04 - 20
# image_05 - 0
# image_06 - 100

# Feature Density Score:

# This score represents how densely features (corners) are distributed across the image.
# Calculated by dividing the number of detected corners by the maximum number of corners expected and then normalizing it to a scale of 0 to 100.
# A higher feature density score indicates that there are many corners detected, suggesting a rich feature set in the image.


# Distribution Score:

# This score measures how evenly distributed the detected corners are across the image.
# Achieved by dividing the image into a grid and counting the number of corners in each grid cell.
# The count is normalized and converted to a percentage to represent how much of the grid is covered by corners.
# A higher distribution score indicates that corners are evenly spread across the image, which is desirable for robust feature detection and tracking.


# Quality Score:

# This score evaluates the quality or strength of the detected corners.
# In this implementation, the quality score is calculated based on the average minimum eigenvalue of corners.
# The minimum eigenvalue represents the corner strength, with higher values indicating stronger corners.
# The average of minimum eigenvalues across all corners is calculated and normalized to a scale of 0 to 100.
# A higher quality score suggests that the detected corners are robust and well-defined.



# Distinctiveness Score:

# This score assesses how distinct or unique the detected features are from each other.
# Typically evaluated using feature descriptors like SIFT or ORB.
# In this implementation, SIFT keypoints are used, and the distinctiveness score is calculated based on the number of keypoints detected.
# The count of keypoints is normalized to a scale of 0 to 100, representing the distinctiveness of features.
# A higher distinctiveness score indicates that the detected features are unique and can be reliably matched.
# Each of these scores provides valuable insights into the characteristics of the detected features in the image, contributing to an overall assessment of trackability. Combining these scores allows for a comprehensive evaluation of image trackability, which is essential for tasks like feature detection, tracking, and augmented reality applications.





