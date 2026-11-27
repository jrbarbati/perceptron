from typing import List


class InvalidInputError(Exception):
    pass


class Perceptron:
    def __init__(self, num_inputs: float | int):
        self.num_inputs = num_inputs
        self.weights = [0] * (num_inputs+1)

    def train(self, data: List[List[float | int]], expected: List[float | int]):
        iterations = 0

        while True:
            iterations += 1
            updated = False

            for training_input, expected_output in zip(data, expected):
                if len(training_input) != self.num_inputs:
                    raise InvalidInputError

                training_input = [1] + training_input

                dp = dot_product(self.weights, training_input)

                actual_output = 1 if dp > 0 else -1

                if expected_output != actual_output:
                    updated = True
                    self._update_weights(training_input, expected_output)

            if not updated:
                print(f'After {iterations} training iterations - Weights: {self.weights}')
                break


    def predict(self, data: List[List[float | int]]):
        if len(data) != self.num_inputs:
            raise InvalidInputError

        data = [1] + data

        return 1 if dot_product(self.weights, data) > 0 else -1


    def _update_weights(self, training_input: List[float | int], expected_output: int):
        self.weights = [
            w + expected_output * t
            for w, t in zip(self.weights, training_input)
        ]


def dot_product(w: List[float | int], x: List[float | int]):
    return sum(w_i * x_i for w_i, x_i in zip(w, x))
