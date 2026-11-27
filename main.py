from typing import List

from sklearn.datasets import load_iris, load_breast_cancer

from perceptron import Perceptron


def _find(data: List, op) -> List:
    result = []

    for datum in data:
        for i, value in enumerate(datum):
            if len(result) < i + 1:
                result.append(value)

            result[i] = op(value, result[i])

    return result


def _find_minimums(data: List) -> List:
    return _find(data, min)


def _find_maximums(data: List) -> List:
    return _find(data, max)


def train_and_predict(data: List[List[float | int]], expectations: List[float | int], num_inputs: int = 2, test_data: List | None = None, test_expectations: List | None = None) -> None:
    p = Perceptron(num_inputs)

    p.train(data, expectations)

    total = 0
    correct = 0

    for i, pred in enumerate(test_data or data):
        actual = p.predict(pred)
        expected = (test_expectations or expectations)[i]

        total += 1
        if actual == expected:
            correct += 1

        print(f"predict({pred}) = {actual}\t Label: {expected}")
    print(f'Accuracy: {(correct / total) * 100}%')


def train_and_predict_logical_and():
    print("Training logical and")

    training_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expectations = [-1, -1, -1, 1]

    train_and_predict(training_data, expectations)


def train_and_predict_logical_or():
    print("Training logical or")

    training_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expectations = [-1, 1, 1, 1]

    train_and_predict(training_data, expectations)


def train_and_predict_logical_not():
    print("Training logical not")

    training_data = [[0], [1]]
    expectations = [1, -1]

    train_and_predict(training_data, expectations, num_inputs=1)


def train_and_predict_iris():
    print("Training iris")

    iris = load_iris()
    x = iris.data
    y = [-1 if y == 0 else 1 for y in (iris.target == 0).astype(int)]

    training_data = [
        x_i.tolist()
        for i, x_i in enumerate(x)
        if 0 <= i%50 < 10
    ]
    expectations = [
        y_i
        for i, y_i in enumerate(y)
        if 0 <= i%50 < 10
    ]

    testing_data = [
        x_i.tolist()
        for i, x_i in enumerate(x)
        if 10 <= i%50 < 50
    ]
    test_expectations = [
        y_i
        for i, y_i in enumerate(y)
        if 10 <= i%50 < 50
    ]

    train_and_predict(
        training_data,
        expectations,
        num_inputs=4,
        test_data=testing_data,
        test_expectations=test_expectations
    )


def train_and_predict_breast_cancer():
    print("Training breast cancer")

    breast_cancer = load_breast_cancer()
    x = breast_cancer.data
    y = [-1 if y == 0 else 1 for y in (breast_cancer.target == 0).astype(int)]

    training_data = [x_i.tolist() for x_i in x[:100]]
    expectations = y[:100]

    mins = _find_minimums(training_data)
    maxes = _find_maximums(training_data)

    # Normalize training data to reduce epochs
    training_data = [
        [(x_i_j-min_) / (max_-min_) for x_i_j, min_, max_ in zip(x_i, mins, maxes)]
        for x_i in training_data
    ]

    testing_data = [x_i.tolist() for x_i in x[100:]]
    # Normalize testing data using the mins and maxes from training so as not to leak testing information into the training set
    testing_data = [
        [(x_i_j - min_) / (max_ - min_) for x_i_j, min_, max_ in zip(x_i, mins, maxes)]
        for x_i in testing_data
    ]
    test_expectations = y[100:]

    train_and_predict(
        training_data,
        expectations,
        num_inputs=30,
        test_data=testing_data,
        test_expectations=test_expectations
    )


def main():
    train_and_predict_logical_and()
    print()
    train_and_predict_logical_or()
    print()
    train_and_predict_logical_not()
    print()
    train_and_predict_iris()
    print()
    train_and_predict_breast_cancer()


if __name__ == "__main__":
    main()
