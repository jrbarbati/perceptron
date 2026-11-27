from typing import List

from perceptron import Perceptron


def train_and_predict(data: List[List[float | int]], expectations: List[float | int], num_inputs: int = 2) -> None:
    p = Perceptron(num_inputs)

    p.train(data, expectations)

    for pred in data:
        actual = p.predict(pred)
        print(f"predict({pred}) = {actual}")


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


def main():
    train_and_predict_logical_and()
    print()
    train_and_predict_logical_or()
    print()
    train_and_predict_logical_not()


if __name__ == "__main__":
    main()
