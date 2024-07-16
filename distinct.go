// package main

// import (
// 	"fmt"
// 	"math"
// 	"math/rand"
// 	"time"

// 	"gocv.io/x/gocv"
// )

// // detectAndCompute detects keypoints and computes descriptors using the provided detector
// func detectAndCompute(image gocv.Mat, detector gocv.SIFT) ([]gocv.KeyPoint, gocv.Mat) {
// 	keypoints, descriptors := detector.DetectAndCompute(image, gocv.NewMat())
// 	return keypoints, descriptors
// }

// // calculateUniquenessScore calculates the uniqueness score for the given descriptors
// func calculateUniquenessScore(descriptors gocv.Mat) float64 {
// 	if descriptors.Empty() {
// 		return 0
// 	}

// 	rows := descriptors.Rows()
// 	cols := descriptors.Cols()
// 	distances := make([]float64, rows)

// 	for i := 0; i < rows; i++ {
// 		minDist := math.Inf(1)
// 		for j := 0; j < rows; j++ {
// 			if i != j {
// 				dist := euclideanDistance(descriptors.Row(i), descriptors.Row(j), cols)
// 				if dist < minDist {
// 					minDist = dist
// 				}
// 			}
// 		}
// 		distances[i] = minDist
// 	}

// 	uniquenessScore := mean(distances)
// 	return uniquenessScore
// }

// // euclideanDistance calculates the Euclidean distance between two rows of descriptors
// func euclideanDistance(rowA, rowB gocv.Mat, length int) float64 {
// 	sum := 0.0
// 	for i := 0; i < length; i++ {
// 		diff := float64(rowA.GetUCharAt(0, i)) - float64(rowB.GetUCharAt(0, i))
// 		sum += diff * diff
// 	}
// 	return math.Sqrt(sum)
// }

// // calculateKeypointStrength calculates the average strength of keypoints
// func calculateKeypointStrength(keypoints []gocv.KeyPoint) float64 {
// 	if len(keypoints) == 0 {
// 		return 0
// 	}
// 	sum := 0.0
// 	for _, kp := range keypoints {
// 		sum += float64(kp.Response)
// 	}
// 	return sum / float64(len(keypoints))
// }

// // calculateRepeatabilityScore calculates the repeatability score between keypoints and transformed keypoints
// func calculateRepeatabilityScore(keypoints, transformedKeypoints []gocv.KeyPoint) float64 {
// 	if len(keypoints) == 0 {
// 		return 0
// 	}
// 	intersection := intersectKeypoints(keypoints, transformedKeypoints)
// 	return float64(len(intersection)) / float64(len(keypoints))
// }

// // intersectKeypoints finds the intersection of two slices of keypoints
// func intersectKeypoints(a, b []gocv.KeyPoint) []gocv.KeyPoint {
// 	intersection := []gocv.KeyPoint{}
// 	for _, kpA := range a {
// 		for _, kpB := range b {
// 			if kpA.ClassID == kpB.ClassID && kpA.X == kpB.Y {
// 				intersection = append(intersection, kpA)
// 				break
// 			}
// 		}
// 	}
// 	return intersection
// }

// // calculateDescriptorDiversity calculates the diversity score of descriptors
// func calculateDescriptorDiversity(descriptors gocv.Mat) float64 {
// 	if descriptors.Empty() {
// 		return 0
// 	}
// 	cols := descriptors.Cols()
// 	variances := make([]float64, cols)
// 	for i := 0; i < cols; i++ {
// 		column := descriptors.Col(i)
// 		variances[i] = variance(column)
// 	}
// 	return mean(variances)
// }

// // variance calculates the variance of a matrix column
// func variance(column gocv.Mat) float64 {
// 	columnFloat64 := matToFloat64Slice(column)
// 	meanVal := mean(columnFloat64)
// 	sum := 0.0
// 	for _, value := range columnFloat64 {
// 		diff := value - meanVal
// 		sum += diff * diff
// 	}
// 	return sum / float64(len(columnFloat64))
// }

// // matToFloat64Slice converts a matrix column to a slice of float64
// func matToFloat64Slice(column gocv.Mat) []float64 {
// 	data := make([]float64, column.Rows())
// 	for i := 0; i < column.Rows(); i++ {
// 		data[i] = float64(column.GetUCharAt(i, 0))
// 	}
// 	return data
// }

// // calculateMatchingAccuracy calculates the matching accuracy score between keypoints and matched keypoints
// func calculateMatchingAccuracy(keypoints, matchedKeypoints []gocv.KeyPoint) float64 {
// 	if len(keypoints) == 0 {
// 		return 0
// 	}
// 	intersection := intersectKeypoints(keypoints, matchedKeypoints)
// 	return float64(len(intersection)) / float64(len(keypoints))
// }

// // calculateSpatialDistribution calculates the spatial distribution score of keypoints
// func calculateSpatialDistribution(keypoints []gocv.KeyPoint, imageSize [2]int) float64 {
// 	if len(keypoints) == 0 {
// 		return 0
// 	}
// 	width := float64(imageSize[1])
// 	height := float64(imageSize[0])

// 	xCoords := make([]float64, len(keypoints))
// 	yCoords := make([]float64, len(keypoints))

// 	for i, kp := range keypoints {
// 		xCoords[i] = float64(kp.X)
// 		yCoords[i] = float64(kp.Y)
// 	}

// 	xHist := histogram(xCoords, 10, 0, width)
// 	yHist := histogram(yCoords, 10, 0, height)

// 	spatialDistributionScore := entropy(xHist) + entropy(yHist)
// 	return spatialDistributionScore
// }

// // histogram creates a histogram from data with a specified number of bins and range
// func histogram(data []float64, bins int, min, max float64) []float64 {
// 	hist := make([]float64, bins)
// 	binSize := (max - min) / float64(bins)

// 	for _, val := range data {
// 		if val >= min && val < max {
// 			bin := int((val - min) / binSize)
// 			hist[bin]++
// 		}
// 	}

// 	return hist
// }

// // entropy calculates the entropy of a histogram
// func entropy(hist []float64) float64 {
// 	sum := 0.0
// 	for _, val := range hist {
// 		sum += val
// 	}
// 	ent := 0.0
// 	for _, val := range hist {
// 		if val > 0 {
// 			p := val / sum
// 			ent -= p * math.Log2(p)
// 		}
// 	}
// 	return ent
// }

// // mean calculates the mean value of a slice of floats
// func mean(values []float64) float64 {
// 	sum := 0.0
// 	for _, value := range values {
// 		sum += value
// 	}
// 	return sum / float64(len(values))
// }

// // calculateDistinctivenessScore calculates the distinctiveness score for the given image
// func calculateDistinctivenessScore(image gocv.Mat, detector gocv.SIFT, transformedImage *gocv.Mat, matchedKeypoints []gocv.KeyPoint) float64 {
// 	keypoints, descriptors := detectAndCompute(image, detector)

// 	if len(keypoints) == 0 || descriptors.Empty() {
// 		return 0
// 	}

// 	uniqueness := calculateUniquenessScore(descriptors)
// 	strength := calculateKeypointStrength(keypoints)
// 	diversity := calculateDescriptorDiversity(descriptors)
// 	spatialDistribution := calculateSpatialDistribution(keypoints, [2]int{image.Rows(), image.Cols()})

// 	repeatability := 1.0
// 	if transformedImage != nil {
// 		transformedKeypoints, _ := detectAndCompute(*transformedImage, detector)
// 		repeatability = calculateRepeatabilityScore(keypoints, transformedKeypoints)
// 	}

// 	matchingAccuracy := 1.0
// 	if matchedKeypoints != nil {
// 		matchingAccuracy = calculateMatchingAccuracy(keypoints, matchedKeypoints)
// 	}

// 	uniquenessNormalized := math.Min(uniqueness/10.0, 1.0) * 100
// 	strengthNormalized := math.Min(strength/0.1, 1.0) * 100
// 	diversityNormalized := math.Min(diversity/0.01, 1.0) * 100
// 	spatialDistributionNormalized := math.Min(spatialDistribution/2.0, 1.0) * 100
// 	repeatabilityNormalized := repeatability * 100
// 	matchingAccuracyNormalized := matchingAccuracy * 100

// 	weights := map[string]float64{
// 		"uniqueness":           0.3,
// 		"strength":             0.2,
// 		"diversity":            0.3,
// 		"spatial_distribution": 0.1,
// 		"repeatability":        0.05,
// 		"matching_accuracy":    0.05,
// 	}

// 	combinedScore := weights["uniqueness"]*uniquenessNormalized +
// 		weights["strength"]*strengthNormalized +
// 		weights["diversity"]*diversityNormalized +
// 		weights["spatial_distribution"]*spatialDistributionNormalized +
// 		weights["repeatability"]*repeatabilityNormalized +
// 		weights["matching_accuracy"]*matchingAccuracyNormalized

// 	finalDistinctivenessScore := combinedScore // Already scaled to 0-100

// 	return finalDistinctivenessScore
// }

// func main() {

// 	detector := gocv.NewSIFT()
// 	defer detector.Close()
// 	rand.Seed(time.Now().UnixNano())

// 	files := []string{
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

// 	for _, file := range files {
// 		fmt.Println("Processing:", file)
// 		image := gocv.IMRead(file, gocv.IMReadColor)
// 		if image.Empty() {
// 			fmt.Printf("Error: Unable to read %s\n", file)
// 			continue
// 		}
// 		distinctScore := calculateDistinctivenessScore(image, detector, nil, nil)
// 		fmt.Printf("Image Distinctiveness Score: %.2f\n", distinctScore)
// 	}
// }
