import random
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Individual(ABC):
    def __init__(self, solution):
        self.solution = solution

    @abstractmethod
    def fitness(self, tsp_data):
        pass