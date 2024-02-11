import random
import math
import copy
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from classIndividual import *
from classPopulation import *

class EvolutionaryAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, offsprings):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.offsprings = offsprings
    
    def random_selection(self, fitness_scores):
        random_number = random.randint(0, len(fitness_scores)-1)
        return random_number

    def truncation_selection_max(self, fitness_scores):
        max_value = max(fitness_scores)
        max_index = fitness_scores.index(max_value)
        return max_index

    def truncation_selection_min(self, fitness_scores):
        min_index = min(fitness_scores)
        min_index = fitness_scores.index(min_index)
        return min_index

    def fitness_proportional_selection_max(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [fitness / total_fitness for fitness in fitness_scores]
        selected_index = random.choices(range(len(fitness_scores)), weights=probabilities)
        return selected_index[0]

    def fitness_proportional_selection_min(self, fitness_scores):
        inverted_fitness_scores = [1 / fitness for fitness in fitness_scores]
        total_inverted_fitness = sum(inverted_fitness_scores)
        probabilities = [inverted_fitness / total_inverted_fitness for inverted_fitness in inverted_fitness_scores]
        selected_index = random.choices(range(len(fitness_scores)), weights=probabilities)
        return selected_index[0]
    
    def rank_based_selection_max(self, fitness_scores):
        duplicate = copy.deepcopy(fitness_scores)
        dic = {}
        i = len(duplicate)
        while duplicate:
            m = max(duplicate)
            duplicate.remove(m)
            dic[m] = i
            i -= 1
        ranked = []
        for j in range(len(fitness_scores)):
            ranked.append(dic[fitness_scores[j]])
        return self.fitness_proportional_selection_max(ranked)

    def rank_based_selection_min(self, fitness_scores):
        duplicate = copy.deepcopy(fitness_scores)
        dic = {}
        i = len(duplicate)
        while duplicate:
            m = min(duplicate)
            duplicate.remove(m)
            dic[m] = i
            i -= 1
        ranked = []
        for j in range(len(fitness_scores)):
            ranked.append(dic[fitness_scores[j]])
        return self.fitness_proportional_selection_min(ranked)

    def binary_tournament_selection_max(self, fitness_scores):
        r1 = self.random_selection(fitness_scores)
        r2 = self.random_selection(fitness_scores)
        while r1 == r2:
            r2 = self.random_selection(fitness_scores)
        if fitness_scores[r1] > fitness_scores[r2]:
            return r1
        else:
            return r2
    
    def binary_tournament_selection_min(self, fitness_scores):
        r1 = self.random_selection(fitness_scores)
        r2 = self.random_selection(fitness_scores)
        while r1 == r2:
            r2 = self.random_selection(fitness_scores)
        if fitness_scores[r1] < fitness_scores[r2]:
            return r1
        else:
            return r2

    @abstractmethod
    def initialize_population():
        pass
    
    @abstractmethod
    def run():
        pass