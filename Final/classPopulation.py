import random
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from classIndividual import *

class Population:
    def __init__(self, individuals):
        self.individuals = individuals

    def fitness_scores(self, tsp_data):
        return [individual.fitness(tsp_data) for individual in self.individuals]

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abstractmethod
    def mutate(self, solution):
        pass