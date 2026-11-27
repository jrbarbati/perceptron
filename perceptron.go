package main

import (
	"errors"
	"fmt"
)

// Perceptron holds weights for classifying data and updating during training
// w0 = bias, x0 = 1, always
// w0x0 + w1x1 + ... + wnxn <= 0 (w dot x) | output = -1
// w0x0 + w1x1 + ... + wnxn > 0  (w dot x) | output = 1
//
// Weight update:
// w(new) = w(old) + yx
type Perceptron struct {
	NumInputs int
	Weights   []float64
}

func New(numInputs int) *Perceptron {
	return &Perceptron{
		NumInputs: numInputs,
		Weights:   make([]float64, numInputs+1),
	}
}

// Train runs the algorithm for perceptron training as follows:
// Step 1:
//
//	Initialize weights to 0 - Done in New()
//
// Step 2:
//
//	If ywx <= 0, then update weights such that w(new) = w(old) + yx
//	where w(old) is the old weight vector and x is the input vector and y is the expected output (-1 or 1)
//
// Step 3:
//
//	Repeat Step 2 until weight vector is not updated, then terminate.
func (p *Perceptron) Train(trainingData [][]float64, expectedOutput []float64) error {
	var iteration int
	var updated bool

	for {
		iteration++
		updated = false

		for i, trainingPoint := range trainingData {
			if len(trainingPoint) != p.NumInputs {
				return errors.New("training data input length did not match expected input length")
			}

			var output float64

			in := inputs(trainingPoint)

			if dp := dotProduct(p.Weights, in); dp <= 0 {
				output = -1
			} else {
				output = 1
			}

			if output*expectedOutput[i] <= 0 {
				updated = true
				p.updateWeights(expectedOutput[i], in)
			}
		}

		if !updated {
			fmt.Printf("After iteration %v - Weights: %v\n", iteration, p.Weights)
			break
		}
	}

	return nil
}

// Predict will attempt to classify the input data based on the trained weights.
// Assumes p.Train has been run with solid training data
func (p *Perceptron) Predict(input []float64) (float64, error) {
	if len(input) != p.NumInputs {
		return 0, errors.New("input length did not match expected input length")
	}

	if output := dotProduct(p.Weights, inputs(input)); output > 0 {
		return 1.0, nil
	}

	return -1.0, nil
}

func (p *Perceptron) updateWeights(output float64, inputs []float64) {
	for i := range p.Weights {
		p.Weights[i] += output * inputs[i]
	}
}

func inputs(x []float64) []float64 {
	i := make([]float64, len(x)+1)

	i[0] = 1
	for j := range x {
		i[j+1] = x[j]
	}

	return i
}

func dotProduct(w []float64, x []float64) (result float64) {
	for i := range w {
		result += w[i] * x[i]
	}

	return result
}
