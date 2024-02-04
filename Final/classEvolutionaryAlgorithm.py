import random
import math
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
        sorted_list = sorted(fitness_scores, reverse=True)
        indexed_list = list(enumerate(sorted_list, start=1))
        weight_mapping = {rank: fitness for rank, fitness in indexed_list}
        ranks = [index for index, _ in indexed_list]
        total_rank_sum = sum(ranks)
        probabilities = [rank / total_rank_sum for rank in ranks]
        selected_index = random.choices(ranks, weights = probabilities)
        return fitness_scores.index(weight_mapping[selected_index[0]])

    def rank_based_selection_min(self, fitness_scores):
        sorted_list = sorted(fitness_scores)
        indexed_list = list(enumerate(sorted_list, start=1))
        weight_mapping = {rank: fitness for rank, fitness in indexed_list}
        ranks = [index for index, _ in indexed_list]
        # Reverse the ranks to give higher probabilities to lower fitness individuals
        reversed_ranks = list(reversed(ranks))
        total_rank_sum = sum(reversed_ranks)
        probabilities = [rank / total_rank_sum for rank in reversed_ranks]
        selected_rank = random.choices(reversed_ranks, weights=probabilities)[0]
        # Find the corresponding fitness score using the reversed ranks
        selected_index = fitness_scores.index(weight_mapping[selected_rank])
        return selected_index

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
    def initialize_population(self, tsp_data):
        pass

    def run(self, tsp_data, pop):
        for generation in range(self.generations):
            fitness_scores = pop.fitness_scores(tsp_data)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(self.offsprings // 2):
                parent1 = pop.individuals[pop.rank_based_selection(fitness_scores)]
                parent2 = pop.individuals[pop.rank_based_selection(fitness_scores)]
                child1, child2 = pop.crossover(parent1, parent2)
                random_number_1 = random.random()
                random_number_2 = random.random()
                if random_number_1 > self.mutation_rate:
                    offspring.append(child1)
                else:
                    child1 = pop.mutate(child1)
                    offspring.append(child1)
                if random_number_2 > self.mutation_rate:
                    offspring.append(child2)
                else:
                    child2 = pop.mutate(child2)
                    offspring.append(child2)
                    
            for i in offspring:
                pop.individuals.append(i)

            fitness_scores = pop.fitness_scores(tsp_data)
            
            temp_population = []
            for i in range(self.population_size):
                x = pop.truncation_selection_min(fitness_scores)
                y = pop.individuals[x]
                pop.individuals.pop(x)
                fitness_scores.pop(x)
                temp_population.append(y)
            pop.individuals = temp_population
                
            best_solution = min(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            print("Generation", generation, ": Best:",best_solution, "Average:", average_fitness)
                
        best_solution = min(fitness_scores)
        average_fitness = sum(fitness_scores) / len(fitness_scores)
        return pop, pop.individuals[fitness_scores.index(best_solution)], best_solution, average_fitness