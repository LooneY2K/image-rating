import cv2

def compute_trackability_score(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect corners using FAST
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(blurred_image, None)

    if keypoints:
        # Filter keypoints based on response (quality)
        keypoints = [kp for kp in keypoints if kp.response > 0]

        # Compute trackability score based on the number of keypoints
        trackability_score = min(len(keypoints) / 500 * 100, 100)  # Normalize to 0-100 range
        return trackability_score
    else:
        return 0

# Compute the trackability score for the image
files = ["test_01.jpeg", "test_02.jpeg", "test_03.jpeg", "test_04.jpeg", "test_05.jpeg", "test_06.jpeg"]
for file in files:
    print("processing: ", file)
    trackability_score = compute_trackability_score(file)
    print(f"Trackability Score: {trackability_score}")
