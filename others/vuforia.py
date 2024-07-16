import cv2
import numpy as np

def vuforia_like_score(image_path):
    """
    Calculates an image score based on SIFT features, distribution, distinctiveness,
    color aspects, and clarity, similar to Vuforia's approach.
    Args:
        image_path (str): Path to the image file.
    Returns:
        float: A score between 0 and 100 representing the image's suitability for AR tracking.
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Feature detection with SIFT
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        num_keypoints = len(keypoints) if keypoints is not None else 0
        
        # Feature distribution (Grid-based approach)
        grid_size = 20  # Adjust grid size as needed
        height, width = image.shape[:2]
        rows, cols = height // grid_size, width // grid_size
        grid = np.zeros((rows, cols), dtype=np.int32)  # Ensure grid has integer data type
        for kp in keypoints:
            x, y = int(kp.pt[0] // grid_size), int(kp.pt[1] // grid_size)
            if x >= 0 and x < cols and y >= 0 and y < rows:  # Check for valid indices
                grid[y, x] += 1
        grid_nonzero = grid[grid > 0]
        if grid_nonzero.size > 0:
            distribution_score = grid_nonzero.std() / grid_nonzero.max()  # Normalize by standard deviation and maximum
        else:
            distribution_score = 0
        
        # Feature distinctiveness (Harris corner response)
        dst = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.04)
        dst_nonzero = dst[dst > 0.01 * dst.max()]  # Consider only the top 1% of corner response values
        if dst_nonzero.size > 0:
            distinctiveness_score = dst_nonzero.mean() / dst_nonzero.max()  # Normalize by mean and maximum
        else:
            distinctiveness_score = 0
        
        # Color aspects
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        color_variation = np.mean(np.std(lab[:, :, 1:], axis=(0, 1)))  # Measure color channel variation
        color_clarity = np.mean(lab[:, :, 1:].std())  # Measure color channel clarity (a* and b*)
        color_score = (color_variation + color_clarity) / 2  # Combine color aspects
        
        # Image clarity (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        clarity_score = 1 - (blur_score / blur_score.max()) if blur_score.max() > 0 else 1  # Higher score for less blurriness

                # Feature Quality: Use Minimum Eigenvalue to evaluate corner strength
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        min_eigenval = cv2.cornerMinEigenVal(blurred_image, 3)
        quality_score = np.mean(min_eigenval)
        quality_score = min(max(quality_score, 0), 1) * 100
        
        # Calculate and normalize scores (adjust weights as needed)
        max_keypoints = 500
        kp = sift.detect(blurred_image, None)  # Adjust this value based on your requirements
        feature_score = min(len(kp) / max_keypoints, 1) # Normalize and weight features (30% importance)
        distribution_score = distribution_score  # Normalize and weight distribution (20% importance)
        distinctiveness_score = distinctiveness_score  # Normalize and weight distinctiveness (15% importance)
        color_score = min(color_score, 1)  # Normalize and weight color aspects (20% importance)
        quality_score = quality_score
        
        score = (feature_score + distribution_score + distinctiveness_score + color_score + quality_score)
        
        # Print individual scores
        print(f"Feature score: {feature_score * 100:.2f}")
        print(f"Distribution score: {distribution_score * 100:.2f}")
        print(f"Distinctiveness score: {distinctiveness_score * 100:.2f}")
        print(f"Color score: {color_score * 100:.2f}")
        print(f"Quality Score: {quality_score * 100:.2f}")
        
        return score * 100  # Multiply by 100 to get the score out of 100
    except Exception as e:
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    files = ["../Flam/FLANN_SERVER/NewDataset/img1.jpg", "../Flam/FLANN_SERVER/NewDataset/img2.jpg", "../Flam/FLANN_SERVER/NewDataset/img3.jpg", "../Flam/FLANN_SERVER/NewDataset/img4.jpg", "../Flam/FLANN_SERVER/NewDataset/img5.jpg", "../Flam/FLANN_SERVER/NewDataset/img6.jpg"]
    for file in files:
        print("processing: ", file)
      # Replace with your image path
        score = vuforia_like_score(file)
        print(f"Image score: {score:.2f}/100")  # Print the score out of 100
    # Example threshold for good score (adjust based on your needs)
    # if score >= 60:
    #     print("This image is likely suitable for AR tracking based on various parameters.")
    # else:
    #     print("This image might have lower performance in AR tracking. Consider improving features, distribution, distinctiveness, color, or clarity.")