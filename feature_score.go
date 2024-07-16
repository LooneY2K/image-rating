// package main

// import (
// 	"fmt"
// 	"math"

// 	"gocv.io/x/gocv"
// )

// // detectAndCompute detects keypoints and computes descriptors using the provided detector
// func detectAndCompute(image gocv.Mat, detector gocv.SIFT) ([]gocv.KeyPoint, gocv.Mat) {
// 	keypoints, descriptors := detector.DetectAndCompute(image, gocv.NewMat())
// 	return keypoints, descriptors
// }

// // calculateUniquenessScore calculates the uniqueness score for the given descriptors
// func calculateUniquenessScore(descriptors gocv.Mat) float64 {
// 	rows := descriptors.Rows()
// 	if rows == 0 {
// 		return 0
// 	}

// 	cols := descriptors.Cols()
// 	descriptorsData := descriptors.ToBytes()
// 	distances := make([][]float64, rows)
// 	for i := 0; i < rows; i++ {
// 		distances[i] = make([]float64, rows)
// 		for j := 0; j < rows; j++ {
// 			if i != j {
// 				distances[i][j] = euclideanDistance(descriptorsData[i*cols:(i+1)*cols], descriptorsData[j*cols:(j+1)*cols])
// 			} else {
// 				distances[i][j] = math.Inf(1)
// 			}
// 		}
// 	}

// 	minDistances := make([]float64, rows)
// 	for i := 0; i < rows; i++ {
// 		minDistances[i] = min(distances[i])
// 	}

// 	uniquenessScore := mean(minDistances)
// 	return uniquenessScore
// }

// // euclideanDistance calculates the Euclidean distance between two points
// func euclideanDistance(a, b []byte) float64 {
// 	sum := 0.0
// 	for i := 0; i < len(a); i++ {
// 		diff := float64(a[i]) - float64(b[i])
// 		sum += diff * diff
// 	}
// 	return math.Sqrt(sum)
// }

// // min returns the minimum value in a slice of floats
// func min(values []float64) float64 {
// 	minVal := values[0]
// 	for _, value := range values {
// 		if value < minVal {
// 			minVal = value
// 		}
// 	}
// 	return minVal
// }

// // mean returns the mean value of a slice of floats
// func mean(values []float64) float64 {
// 	sum := 0.0
// 	for _, value := range values {
// 		sum += value
// 	}
// 	return sum / float64(len(values))
// }

// // calculateCoverageScore calculates the coverage score for the given keypoints and image dimensions
// func calculateCoverageScore(keypoints []gocv.KeyPoint, image gocv.Mat) float64 {
// 	if len(keypoints) == 0 {
// 		return 0
// 	}
// 	height := float64(image.Rows())
// 	width := float64(image.Cols())

// 	xPoints := make([]float64, len(keypoints))
// 	yPoints := make([]float64, len(keypoints))

// 	for i, kp := range keypoints {
// 		xPoints[i] = float64(kp.X)
// 		yPoints[i] = float64(kp.Y)
// 	}

// 	xCoverage := (max(xPoints) - min(xPoints)) / width
// 	yCoverage := (max(yPoints) - min(yPoints)) / height

// 	coverageScore := (xCoverage + yCoverage) / 2
// 	return coverageScore
// }

// // max returns the maximum value in a slice of floats
// func max(values []float64) float64 {
// 	maxVal := values[0]
// 	for _, value := range values {
// 		if value > maxVal {
// 			maxVal = value
// 		}
// 	}
// 	return maxVal
// }

// // calculateFeatureScore calculates the feature score for the given image
// func calculateFeatureScore(image gocv.Mat) float64 {
// 	detector := gocv.NewSIFT()
// 	defer detector.Close()
// 	keypoints, descriptors := detectAndCompute(image, &detector)
// 	if len(keypoints) == 0 || descriptors.Empty() {
// 		return 0
// 	}

// 	uniqueness := calculateUniquenessScore(descriptors)
// 	coverage := calculateCoverageScore(keypoints, image)
// 	numFeatures := float64(len(keypoints))

// 	// Normalize scores
// 	normalizedUniqueness := math.Min(uniqueness/1000, 1.0) // Adjust normalization factor
// 	normalizedCoverage := math.Min(coverage, 1.0)          // Coverage is naturally between 0 and 1
// 	normalizedNumFeatures := math.Min(numFeatures/1000, 1.0)

// 	// Combine scores with weights
// 	weights := map[string]float64{"uniqueness": 0.4, "coverage": 0.2, "num_features": 0.4}

// 	combinedScore := (weights["uniqueness"]*normalizedUniqueness +
// 		weights["coverage"]*normalizedCoverage +
// 		weights["num_features"]*normalizedNumFeatures) * 100 // Scale to 0-100

// 	return combinedScore
// }

// // processImages processes the images and calculates their feature scores
// func processImages(files []string) {
// 	for _, file := range files {
// 		fmt.Println("Processing:", file)
// 		image := gocv.IMRead(file, gocv.IMReadColor)
// 		if image.Empty() {
// 			fmt.Printf("Error: Unable to read %s\n", file)
// 			continue
// 		}
// 		score := calculateFeatureScore(image)
// 		fmt.Printf("Image Feature Score: %.2f\n", score)
// 	}
// }

// func main() {
// 	files := []string{
// 		"new_images/test_01.png",
// 		"dataset/test_01.png",
// 		"dataset/test_02.png",
// 		"dataset/test_03.png",
// 		"dataset/test_04.png",
// 		"dataset/test_05.png",
// 		"dataset/test_09.png",
// 		"dataset/test_10.png",
// 		"dataset/test_11.png",
// 		"dataset/test_12.png",
// 		"dataset/test_13.png",
// 		"dataset/test_14.png",
// 		"dataset/test_15.png",
// 		"dataset/test_16.png",
// 		"dataset/test_17.png",
// 		"dataset/test_18.png",
// 		"dataset/test_19.png",
// 		"dataset/test_20.png",
// 		"dataset/test_21.png",
// 		"dataset/test_22.png",
// 		"dataset/test_23.png",
// 		"dataset/test_24.png",
// 		"dataset/test_25.png",
// 		"dataset/test_26.png",
// 		"dataset/test_27.png",
// 		"dataset/test_29.png",
// 		"dataset/test_30.png",
// 		"dataset/test_31.png",
// 		"dataset/test_32.png",
// 		"dataset/test_33.png",
// 		"dataset/test_34.png",
// 		"dataset/test_35.png",
// 		"dataset/test_36.png",
// 		"dataset/test_37.png",
// 		"dataset/test_38.png",
// 		"dataset/test_39.png",
// 		"dataset/test_40.png",
// 		"dataset/test_41.png",
// 		"dataset/test_42.png",
// 		"dataset/test_43.png",
// 		"dataset/test_44.png",
// 		"dataset/test_45.png",
// 		"dataset/test_46.png",
// 		"dataset/test_47.png",
// 		"dataset/test_49.png",
// 		"dataset/test_50.png",
// 		"dataset/test_51.png",
// 		"dataset/test_52.png",
// 		"dataset/test_53.png",
// 		"dataset/test_54.png",
// 		"dataset/test_55.png",
// 		"dataset/test_56.png",
// 		"dataset/test_57.png",
// 		"dataset/test_58.png",
// 	}

// 	processImages(files)
// }
