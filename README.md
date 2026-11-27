# Perceptron

Implementation of Rosenblatt's perceptron


Handwritten based on reading of "Why Machines Learn - The Elegant math Behind Modern AI" by Anil Ananthaswamy

This is where the pseudocode in perceptron.Train comes from (with some paraphrasing from me)

The perceptron implementation is testing on 3, somewhat trivial, cases of the logical AND, OR, and NOT operators. The training data is the complete set of 2-input logic to each of the gates (or 1-input for NOT). The prediction (classify) is then run on the same data set. I understand this is bad for testing to test on exactly what you trained on, but for this POC it's fine. See "expansion" note below.

### Expansion

Bring in Python's Scikit-Learn and download a few datasets that are already binary outputs or turn them into binary outputs and train/predict on a non-trivial dataset while properly splitting out the training vs testing data unlike what I did in my logical operator POCs.

### Results
Results are as follows from running main.go:

```
Starting Training of Logical AND
After iteration 6 - Weights: [-2 2 1]
Predict([0 0]) = -1.000000
Predict([0 1]) = -1.000000
Predict([1 0]) = -1.000000
Predict([1 1]) = 1.000000

Starting Training of Logical OR
After iteration 4 - Weights: [0 1 1]
Predict([0 0]) = -1.000000
Predict([0 1]) = 1.000000
Predict([1 0]) = 1.000000
Predict([1 1]) = 1.000000

Starting Training of Logical NOT
After iteration 3 - Weights: [1 -1]
Predict([0]) = 1.000000
Predict([1]) = -1.000000
```