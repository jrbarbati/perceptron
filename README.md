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
After 108 training iterations - Weights: [-15, -0.1543703007518829, 3.701048951048951, 0.8418899741213953, 2.1580454626812307, -5.032197244379993, -0.8426268928316193, 9.340713302091233, 0.8131900428651535, 0.15384615384614697, -0.5944917151813646, 2.096210188072967, -1.1039981511439763, 5.004788507581811, 7.023830021694983, 4.328263804989001, -2.8430737400863064, -0.19867505971468521, -5.811477418125402, -3.8965216412450188, -4.75276109281148, 5.707834189009306, 3.4196523589925523, 8.159349909349938, 8.846763317599494, 3.0104373328732876, 3.1732044553819234, 3.333886598061843, 4.110417649406696, 1.5897890794401641, 4.202807293716393]
Accuracy: 92.75053304904051%
```