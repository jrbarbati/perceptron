package main

import "fmt"

func main() {
	var returnCode int

	returnCode = trainAndPredictLogicalAnd()
	if returnCode != 0 {
		return
	}

	returnCode = trainAndPredictLogicalOr()
	if returnCode != 0 {
		return
	}

	_ = trainAndPredictLogicalNot()
}

func trainAndPredictLogicalAnd() int {
	fmt.Println("Starting Training of Logical AND")

	trainingData := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	expectedOutputs := []float64{
		-1, -1, -1, 1,
	}

	return train(trainingData, expectedOutputs, 2)
}

func trainAndPredictLogicalOr() int {
	fmt.Println("Starting Training of Logical OR")

	trainingData := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	expectedOutputs := []float64{
		-1, 1, 1, 1,
	}

	return train(trainingData, expectedOutputs, 2)
}

func trainAndPredictLogicalNot() int {
	fmt.Println("Starting Training of Logical NOT")

	trainingData := [][]float64{
		{0},
		{1},
	}

	expectedOutputs := []float64{
		-1, 1,
	}

	return train(trainingData, expectedOutputs, 1)
}

func train(data [][]float64, expectations []float64, numInputs int) int {
	p := New(numInputs)

	trainErr := p.Train(data, expectations)

	if trainErr != nil {
		fmt.Println("Error training data:", trainErr)
		return 1
	}

	for _, pred := range data {
		predict, err := p.Predict(pred)

		if err != nil {
			fmt.Println("Error predicting data:", pred, err)
			return 1
		}

		fmt.Printf("Predict(%v) = %f\n", pred, predict)
	}

	fmt.Println("")
	return 0
}
