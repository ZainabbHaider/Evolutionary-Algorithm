import random
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from classIndividual import *
from classPopulation import *
from classEvolutionaryAlgorithm import *
import ast

class TSP_Individual(Individual):
    def __init__(self, solution):
        super().__init__(solution)

    def fitness(self, tsp_data):
        fitness = 0
        for i in range(len(self.solution) - 1):
            x1, y1 = tsp_data[self.solution[i]]
            x2, y2 = tsp_data[self.solution[i + 1]]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            fitness += distance
        return fitness

class TSP_Population(Population):
    def __init__(self, individuals):
        super().__init__(individuals)

    def fitness_scores(self, tsp_data):
        return [individual.fitness(tsp_data) for individual in self.individuals]

    def crossover(self, parent1, parent2):
        # Child 1
        point1, point2 = sorted(random.sample(range(1, len(parent1.solution)), 2))
        child1 = []
        child1_middle = parent2.solution[point1:point2]  # Middle part from parent2
        remaining_num = []
        for i in range(point2, len(parent1.solution)):
            if parent1.solution[i] not in child1_middle:
                remaining_num.append(parent1.solution[i])
        for i in range(point2):
            if parent1.solution[i] not in child1_middle:
                remaining_num.append(parent1.solution[i])
        for i in range(point2, len(parent1.solution)):
            child1_middle.append(remaining_num.pop(0))
        
        for i in range(point1):
            child1.append(remaining_num.pop(0))
        child1 = child1 + child1_middle 
        c1 = TSP_Individual(child1)
        
        # Child 2
        child2 = []
        child2_middle = parent1.solution[point1:point2]  # Middle part from parent1
        remaining_num2 = []
        for i in range(point2, len(parent2.solution)):
            if parent2.solution[i] not in child2_middle:
                remaining_num2.append(parent2.solution[i])
        for i in range(point2):
            if parent2.solution[i] not in child2_middle:
                remaining_num2.append(parent2.solution[i])
        for i in range(point2, len(parent2.solution)):
            child2_middle.append(remaining_num2.pop(0))
        for i in range(point1):
            child2.append(remaining_num2.pop(0))
        child2 = child2 + child2_middle    
        c2 = TSP_Individual(child2)
        
        return c1, c2

    # Perform mutation
    def mutate(self, solution):
        mutated_solution = solution.solution[:]
        random_index_1 = random.randint(0, len(mutated_solution)-1)
        random_index_2 = random.randint(0, len(mutated_solution)-1)
        while random_index_1 == random_index_2:
            random_index_2 = random.randint(0, len(mutated_solution)-1)
            
        if random_index_1 > random_index_2:
            removed_item = mutated_solution.pop(random_index_1)
            mutated_solution.insert((random_index_2+1), removed_item)
        else:
            removed_item = mutated_solution.pop(random_index_2)
            mutated_solution.insert((random_index_1+1), removed_item)

        m = TSP_Individual(mutated_solution)
        return m
    
    def mutate1(self, solution):
        mutated_solution = solution.solution[:]
        random_index_1 = random.randint(0, len(mutated_solution)-1)
        random_index_2 = random.randint(0, len(mutated_solution)-1)
        while random_index_1 == random_index_2:
            random_index_2 = random.randint(0, len(mutated_solution)-1)

        mutated_solution[random_index_1], mutated_solution[random_index_2] = mutated_solution[random_index_2], mutated_solution[random_index_1]

        m = TSP_Individual(mutated_solution)
        return m
    
    def mutate2(self, solution):
        mutated_solution = solution.solution[:]
        
        start, end = sorted(random.sample(range(len(mutated_solution)), 2))
        segment = mutated_solution[start:end]
        del mutated_solution[start:end]
        insertion_point = random.randint(0, len(mutated_solution))
        mutated_solution[insertion_point:insertion_point] = segment
        
        m = TSP_Individual(mutated_solution)
        return m

class TSP_EA(EvolutionaryAlgorithm):
    def __init__(self, population_size, generations, mutation_rate, offsprings, replacement, elite):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.offsprings = offsprings
        self.replacement = replacement
        self.elite = elite

    def initialize_population(self, tsp_data):
        individuals = []
        for i in range(self.population_size):
            all_values = list(tsp_data.keys())
            random.shuffle(all_values)
            individual = TSP_Individual(all_values)
            individuals.append(individual)
        return TSP_Population(individuals)

    def run(self, tsp_data, pop):
        best_fitness_values = []
        avg_fitness_values = []
        
        for generation in range(self.generations):
            fitness_scores = pop.fitness_scores(tsp_data)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(self.offsprings // 2):
                parent1 = pop.individuals[self.binary_tournament_selection_min(fitness_scores)]
                parent2 = pop.individuals[self.binary_tournament_selection_min(fitness_scores)]
                child1, child2 = pop.crossover(parent1, parent2)
                random_number_1 = random.random()
                random_number_2 = random.random()
                if random_number_1 > self.mutation_rate:
                    offspring.append(child1)
                else:
                    child1 = pop.mutate2(child1)
                    offspring.append(child1)
                if random_number_2 > self.mutation_rate:
                    offspring.append(child2)
                else:
                    child2 = pop.mutate2(child2)
                    offspring.append(child2)
            
            if self.replacement:
                # replacement implementation
                pop.individuals = offspring
            else:
                # regular implementation
                for i in offspring:
                    pop.individuals.append(i)
            
            if self.elite:
                file_path = "HallofFame.txt"
                elite_data = []
                with open(file_path, 'r') as file:
                    for line in file:
                        line = line.strip()  # Remove leading/trailing whitespaces and newline characters
                        if line:  # Ensure the line is not empty
                            items = line.split(';')  # Split the line by ';'
                            sol = ast.literal_eval(items[0])
                            sol = TSP_Individual(sol)
                            elite_data.append(sol)  # Append the extracted information as a tuple
                
                # add elite individuals to population
                r = random.random()
                if r < 0.7:
                    for i in elite_data:
                        if i not in pop.individuals:
                            r2 = random.random()
                            if r2 < 0.5:
                                pop.individuals.append(i)

            fitness_scores = pop.fitness_scores(tsp_data)
            
            temp_population = []
            for i in range(self.population_size):
                x = self.truncation_selection_min(fitness_scores)
                y = pop.individuals[x]
                pop.individuals.pop(x)
                fitness_scores.pop(x)
                temp_population.append(y)
            pop.individuals = temp_population
            fitness_scores = pop.fitness_scores(tsp_data)
                
            best_solution = min(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            
            best_fitness_values.append(best_solution)
            avg_fitness_values.append(average_fitness)
            
            print("Generation", generation+1, ": Best:",best_solution, "Average:", average_fitness)

        best_solution = min(fitness_scores)
        return pop, pop.individuals[fitness_scores.index(best_solution)], best_solution, best_fitness_values, avg_fitness_values