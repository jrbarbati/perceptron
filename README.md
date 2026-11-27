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

### Python Results
After translating the algorithm into Python to unlock the sklearn data sets without doing weird c-stuff in both Go and Python, I have some interesting results from 2 more data sets - the Iris and Breast Cancer data sets. 

Iris needed some preprocessing to get it into a binary-shape (one vs the rest). I also, somewhat arbitrarily, decided on training on the first 10 of each petal type (30 total) and then test on the rest (120).

Breast Cancer didn't need preprocessing, but normalizing the training and test data reduced the epochs from ~830,000 to 108 and increased accuracy from 91.897654% to ~92.75%. Training consisted of the first 100 data-points of 569 in the set and that number can be tweaked as I decided somewhat arbitrarily.

```
Training logical and
After 6 training iterations - Weights: [-2, 2, 1]
predict([0, 0]) = -1	 Label: -1
predict([0, 1]) = -1	 Label: -1
predict([1, 0]) = -1	 Label: -1
predict([1, 1]) = 1	 Label: 1
Accuracy: 100.0%

Training logical or
After 4 training iterations - Weights: [0, 1, 1]
predict([0, 0]) = -1	 Label: -1
predict([0, 1]) = 1	 Label: 1
predict([1, 0]) = 1	 Label: 1
predict([1, 1]) = 1	 Label: 1
Accuracy: 100.0%

Training logical not
After 3 training iterations - Weights: [1, -1]
predict([0]) = 1	 Label: 1
predict([1]) = -1	 Label: -1
Accuracy: 100.0%

Training iris
After 4 training iterations - Weights: [1, 1.299999999999999, 4.1, -5.200000000000001, -2.1999999999999997]
Accuracy: 100.0%

Training breast cancer
After 4801 training iterations - Weights: [-58, -1.9571205452217508, 0.42103483260105135, -8.60845829590286, 26.139469777304583, 29.935090728535112, -63.329771933437605, 3.654475164011322, 60.25539264412482, -33.63288009888501, 12.58353352260386, 115.62552960347668, -23.010342998593273, 26.045780521134716, 47.646289302521474, -19.81425706224151, 13.556080451828501, -47.288338383844035, 55.73025194168256, -21.77669921611705, -43.19533463233548, 15.55852009961187, 38.86140724947399, -24.762189352058908, 32.5575353912758, 8.874001188667572, -32.466390661212095, 80.82497435898829, -11.216632302404996, 57.64662547527345, 8.849234393402856]
Accuracy: 93.0%
```