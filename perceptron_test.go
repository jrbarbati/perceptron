package main

import (
	"math/rand/v2"
	"testing"
)

func TestInputs(t *testing.T) {
	scenarios := []struct {
		name     string
		x        []float64
		expected []float64
	}{
		{
			name:     "zero",
			x:        []float64{},
			expected: []float64{1},
		},
		{
			name:     "one",
			x:        []float64{3, 2, 0, 13},
			expected: []float64{1, 3, 2, 0, 13},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			output := inputs(scenario.x)

			if len(scenario.expected) != len(output) {
				t.Errorf("Expected %v, got %v", scenario.expected, output)
			}

			for i := range scenario.expected {
				if scenario.expected[i] != output[i] {
					t.Errorf("Expected %v, got %v", scenario.expected, output)
					break
				}
			}
		})
	}
}

func TestDotProduct(t *testing.T) {
	scenarios := []struct {
		name     string
		w        []float64
		x        []float64
		expected float64
	}{
		{
			name:     "zero",
			w:        []float64{0, 0, 0, 0},
			x:        []float64{1, 1, 1, 1},
			expected: 0,
		},
		{
			name:     "one",
			w:        []float64{0.5, 0, 0, 0},
			x:        []float64{1, 1, 1, 1},
			expected: 0.5,
		},
		{
			name:     "two",
			w:        []float64{0.5, 0.25, 0, 0},
			x:        []float64{1, 4, 1, 1},
			expected: 1.5,
		},
		{
			name:     "three",
			w:        []float64{0.5, 0.25, 0.4, 0},
			x:        []float64{1, 4, 2, 10},
			expected: 2.3,
		},
		{
			name:     "four",
			w:        []float64{0.5, 0.25, 0.4, 5},
			x:        []float64{1, 4, 2, 0.1},
			expected: 2.8,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			output := dotProduct(scenario.w, scenario.x)

			if scenario.expected != output {
				t.Errorf("expected: %f, got: %f", scenario.expected, output)
			}
		})
	}
}

func BenchmarkInputs(b *testing.B) {
	length := 100000

	x := make([]float64, length)

	for i := range x {
		x[i] = rand.Float64()
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		inputs(x)
	}
}

func BenchmarkDotProduct(b *testing.B) {
	length := 100000

	w := make([]float64, length)
	x := make([]float64, length)

	for i := range w {
		w[i] = rand.Float64()
		x[i] = rand.Float64()
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		dotProduct(w, x)
	}
}
