from typing import List

from sklearn.datasets import load_iris

from perceptron import Perceptron


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


def main():
    train_and_predict_logical_and()
    print()
    train_and_predict_logical_or()
    print()
    train_and_predict_logical_not()
    print()
    train_and_predict_iris()


if __name__ == "__main__":
    main()
