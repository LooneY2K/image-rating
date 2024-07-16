package main

import (
	"math"

	"gocv.io/x/gocv"
)

// Function to calculate sharpness using the variance of the Laplacian
func calculateSharpness(image gocv.Mat) float64 {
	gray := gocv.NewMat()
	gocv.CvtColor(image, &gray, gocv.ColorBGRToGray)
	laplacian := gocv.NewMat()
	gocv.Laplacian(gray, &laplacian, gocv.MatTypeCV64F, 1, 1, 0, gocv.BorderDefault)
	defer gray.Close()
	defer laplacian.Close()

	mean, stddev := gocv.NewMat(), gocv.NewMat()
	defer mean.Close()
	defer stddev.Close()

	gocv.MeanStdDev(laplacian, &mean, &stddev)
	stddevValue := stddev.GetDoubleAt(0, 0)
	return stddevValue * stddevValue // Variance is the square of the standard deviation
}

// Function to calculate global contrast
func calculateGlobalContrast(image gocv.Mat) float64 {
	gray := gocv.NewMat()
	gocv.CvtColor(image, &gray, gocv.ColorBGRToGray)
	defer gray.Close()

	mean, stddev := gocv.NewMat(), gocv.NewMat()
	defer mean.Close()
	defer stddev.Close()

	gocv.MeanStdDev(gray, &mean, &stddev)
	return stddev.GetDoubleAt(0, 0)
}

// Function to calculate colorfulness
func calculateColorfulness(image gocv.Mat) float64 {
	bgr := gocv.Split(image)
	defer bgr[0].Close()
	defer bgr[1].Close()
	defer bgr[2].Close()

	rg := gocv.NewMat()
	gocv.AbsDiff(bgr[2], bgr[1], &rg)
	yb := gocv.NewMat()
	temp := gocv.NewMat()
	gocv.AddWeighted(bgr[2], 0.5, bgr[1], 0.5, 0, &temp)
	gocv.AbsDiff(temp, bgr[0], &yb)
	defer rg.Close()
	defer yb.Close()
	defer temp.Close()

	meanRg, stddevRg := gocv.NewMat(), gocv.NewMat()
	defer meanRg.Close()
	defer stddevRg.Close()
	gocv.MeanStdDev(rg, &meanRg, &stddevRg)

	meanYb, stddevYb := gocv.NewMat(), gocv.NewMat()
	defer meanYb.Close()
	defer stddevYb.Close()
	gocv.MeanStdDev(yb, &meanYb, &stddevYb)

	colorfulness := math.Sqrt(math.Pow(stddevRg.GetDoubleAt(0, 0), 2)+math.Pow(stddevYb.GetDoubleAt(0, 0), 2)) +
		0.3*math.Sqrt(math.Pow(meanRg.GetDoubleAt(0, 0), 2)+math.Pow(meanYb.GetDoubleAt(0, 0), 2))

	return colorfulness
}

// Function to calculate noise level using a simple threshold method
func calculateNoiseLevel(image gocv.Mat) float64 {
	gray := gocv.NewMat()
	gocv.CvtColor(image, &gray, gocv.ColorBGRToGray)
	defer gray.Close()

	mean, stddev := gocv.NewMat(), gocv.NewMat()
	defer mean.Close()
	defer stddev.Close()

	gocv.MeanStdDev(gray, &mean, &stddev)
	return mean.GetDoubleAt(0, 0) + 2*stddev.GetDoubleAt(0, 0)
}

// Function to calculate the overall quality score
func calculateQualityScore(image gocv.Mat) float64 {
	// Calculate individual quality metrics
	sharpness := calculateSharpness(image)
	contrast := calculateGlobalContrast(image)
	colorfulness := calculateColorfulness(image)
	noiseLevel := calculateNoiseLevel(image)

	// Normalize scores (example normalization factors)
	sharpnessNormalized := math.Min(sharpness/1000.0, 1.0)
	contrastNormalized := math.Min(contrast/128.0, 1.0)
	colorfulnessNormalized := math.Min(colorfulness/100.0, 1.0)
	noiseLevelNormalized := 1.0 - math.Min(noiseLevel/255.0, 1.0)

	// Combine scores with weights
	weights := map[string]float64{
		"sharpness":    0.25,
		"contrast":     0.25,
		"colorfulness": 0.25,
		"noise_level":  0.25,
	}

	combinedScore := weights["sharpness"]*sharpnessNormalized +
		weights["contrast"]*contrastNormalized +
		weights["colorfulness"]*colorfulnessNormalized +
		weights["noise_level"]*noiseLevelNormalized

	finalQualityScore := combinedScore * 100 // Scale to 0-100

	return finalQualityScore
}
