from collections import Counter
from typing import Callable, Any

import numpy as np
from IPython.core.display import Math
from numpy import float64, floating
from numpy._typing import _128Bit, _16Bit, _32Bit, _64Bit, _96Bit
from sympy.tensor.array import ndim_array


class  KNearestNeighbor:

    def __init__(self):
        self.y_train = None
        self.x_train = None
        self.distances: list = []

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    @staticmethod
    def l1_distance(item1, item2) -> float:
        return np.sum(np.abs(item1 - item2))

    @staticmethod
    def l2_distance(item1, item2) -> float:
        return np.sqrt(np.sum((item1 - item2) ** 2))

    @staticmethod
    def l3_distance(item1, item2) -> float:
        return np.cbrt(np.sum(np.abs((item1 - item2)) ** 3))

    @staticmethod
    def ln_distance(item1, item2, order: int) -> float:
        return float(np.linalg.norm(item1 - item2, ord=order).item())

    def calculate_distances(self, x_test, distance_method: Callable[[np.ndarray, np.ndarray], float]) -> None:
        distance_matrix = []

        for i in range(len(x_test)):
            distances_for_this_test_point = []
            for j in range(len(self.x_train)):
                dist: float = distance_method(x_test[i], self.x_train[j])
                distances_for_this_test_point.append((dist, self.y_train[j]))

            distance_matrix.append(distances_for_this_test_point)
            print(f"{i}/{str(len(x_test))}")

        self.distances = distance_matrix

    def predict(self, k) -> list:
          predicted_labels = []
          for i in range(len(self.distances)):
            distances_for_this_test_point = self.distances[i]
            distances_for_this_test_point.sort(key=lambda x: x[0])
            k_nearest = distances_for_this_test_point[:k]
            k_labels = [label for (dist, label) in k_nearest]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predicted_labels.append(most_common)
          return predicted_labels
