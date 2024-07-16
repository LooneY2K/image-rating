package main

import (
	"fmt"
	"log"
	"math"

	"gocv.io/x/gocv"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// detectAndCompute detects keypoints and computes descriptors using the provided detector
func detectAndCompute(image gocv.Mat, detector gocv.SIFT) ([]gocv.KeyPoint, gocv.Mat) {
	keypoints, descriptors := detector.DetectAndCompute(image, gocv.NewMat())
	return keypoints, descriptors
}

// Function to calculate spatial distribution score
func calculateSpatialDistribution(keypoints []gocv.KeyPoint, imageSize []int) float64 {
	if len(keypoints) < 2 {
		return 0 // Return 0 if there are not enough keypoints
	}

	xCoords := make([]float64, len(keypoints))
	yCoords := make([]float64, len(keypoints))
	for i, kp := range keypoints {
		xCoords[i] = float64(kp.X)
		yCoords[i] = float64(kp.Y)
	}
	numBins := 10
	xHist := make([]float64, numBins)
	yHist := make([]float64, numBins)
	xBins := make([]float64, numBins+1)
	yBins := make([]float64, numBins+1)
	for i := 0; i <= numBins; i++ {
		xBins[i] = float64(i) * float64(imageSize[1]) / float64(numBins)
		yBins[i] = float64(i) * float64(imageSize[0]) / float64(numBins)
	}

	stat.Histogram(xHist, xCoords, nil, xBins)
	stat.Histogram(yHist, yCoords, nil, yBins)
	spatialDistributionScore := stat.Entropy(xHist) + stat.Entropy(yHist)
	return spatialDistributionScore
}

// Function to calculate descriptor uniqueness score
func calculateUniquenessScore(descriptors gocv.Mat) float64 {
	rows := descriptors.Rows()
	if rows == 0 {
		return 0 // Return 0 if there are no descriptors
	}

	data := make([]float64, rows*rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < rows; j++ {
			if i != j {
				rowI := descriptors.RowRange(i, i+1)
				rowJ := descriptors.RowRange(j, j+1)
				diffMat := gocv.NewMat()
				gocv.Subtract(rowI, rowJ, &diffMat)
				distance := gocv.Norm(diffMat, gocv.NormL2)
				diffMat.Close()
				rowI.Close()
				rowJ.Close()
				data[i*rows+j] = distance
			} else {
				data[i*rows+j] = math.MaxFloat64
			}
		}
	}
	distances := mat.NewDense(rows, rows, data)
	minDistances := make([]float64, rows)
	for i := 0; i < rows; i++ {
		minDistances[i] = mat.Min(distances.RowView(i))
	}
	uniquenessScore := stat.Mean(minDistances, nil)
	return uniquenessScore
}

// Function to calculate coverage score
func calculateCoverage(keypoints []gocv.KeyPoint, imageSize []int) float64 {
	height, width := imageSize[0], imageSize[1]
	gridSize := 10
	grid := make([][]bool, gridSize)
	for i := range grid {
		grid[i] = make([]bool, gridSize)
	}
	for _, kp := range keypoints {
		x, y := int(kp.X*float64(gridSize)/float64(width)), int(kp.Y*float64(gridSize)/float64(height))
		if x >= 0 && x < gridSize && y >= 0 && y < gridSize {
			grid[y][x] = true
		}
	}
	coverageScore := 0
	for _, row := range grid {
		for _, covered := range row {
			if covered {
				coverageScore++
			}
		}
	}
	return float64(coverageScore) / float64(gridSize*gridSize)
}

// Function to calculate entropy of keypoints
func calculateKeypointEntropy(keypoints []gocv.KeyPoint, imageSize []int) float64 {
	gridSize := 10
	hist := make([]float64, gridSize*gridSize)
	for _, kp := range keypoints {
		x, y := int(kp.X*float64(gridSize)/float64(imageSize[1])), int(kp.Y*float64(gridSize)/float64(imageSize[0]))
		if x >= 0 && x < gridSize && y >= 0 && y < gridSize {
			hist[y*gridSize+x]++
		}
	}
	keypointEntropy := stat.Entropy(hist)
	return keypointEntropy
}

// Function to calculate grid-based uniformity score
func calculateGridUniformity(keypoints []gocv.KeyPoint, imageSize []int) float64 {
	height, width := imageSize[0], imageSize[1]
	gridSize := 10
	grid := make([]float64, gridSize*gridSize)
	for _, kp := range keypoints {
		x, y := int(kp.X*float64(gridSize)/float64(width)), int(kp.Y*float64(gridSize)/float64(height))
		if x >= 0 && x < gridSize && y >= 0 && y < gridSize {
			grid[y*gridSize+x]++
		}
	}

	gridMean := stat.Mean(grid, nil)
	gridStd := stat.StdDev(grid, nil)

	if gridMean == 0 {
		return 0 // Avoid division by zero
	}

	uniformityScore := 1 - (gridStd / gridMean)
	return uniformityScore
}

// Function to calculate the overall distribution score
func calculateDistributionScore(image gocv.Mat) float64 {
	// Detect and compute descriptors
	detector := gocv.NewSIFT()
	defer detector.Close()
	keypoints, descriptors := detectAndCompute(image, detector)

	// Image size
	imageSize := []int{image.Rows(), image.Cols()}

	// Calculate individual distribution metrics
	spatialDistribution := calculateSpatialDistribution(keypoints, imageSize)
	uniqueness := calculateUniquenessScore(descriptors)
	coverage := calculateCoverage(keypoints, imageSize)
	keypointEntropy := calculateKeypointEntropy(keypoints, imageSize)
	gridUniformity := calculateGridUniformity(keypoints, imageSize)

	// Normalize scores (example normalization factors)
	spatialDistributionNormalized := math.Min(spatialDistribution/10.0, 1.0)
	uniquenessNormalized := math.Min(uniqueness/100.0, 1.0)
	coverageNormalized := coverage
	keypointEntropyNormalized := math.Min(keypointEntropy/10.0, 1.0)
	gridUniformityNormalized := gridUniformity

	// Combine scores with weights
	weights := map[string]float64{
		"spatial_distribution": 0.2,
		"uniqueness":           0.2,
		"coverage":             0.2,
		"keypoint_entropy":     0.2,
		"grid_uniformity":      0.2,
	}

	combinedScore := weights["spatial_distribution"]*spatialDistributionNormalized +
		weights["uniqueness"]*uniquenessNormalized +
		weights["coverage"]*coverageNormalized +
		weights["keypoint_entropy"]*keypointEntropyNormalized +
		weights["grid_uniformity"]*gridUniformityNormalized

	finalDistributionScore := combinedScore * 100 // Scale to 0-100

	return finalDistributionScore
}

func main() {
	files := []string{
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
	}

	for _, file := range files {
		fmt.Println("Processing:", file)
		image := gocv.IMRead(file, gocv.IMReadColor)
		if image.Empty() {
			log.Printf("Error: Unable to read %s", file)
			continue
		}
		defer image.Close()

		distributionScore := calculateDistributionScore(image)
		fmt.Printf("Image Distribution Score: %.2f\n", distributionScore)
	}
}
