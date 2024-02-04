import random
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from abc import ABC, abstractmethod
from classIndividual import *
from classPopulation import *
from classEvolutionaryAlgorithm import *

# helper function
def convert_solution_to_schedule(sol, data, num_machines, num_jobs):
    schedule = [] 
    for i in range(num_machines):
        schedule.append([])
    machine_free_at = [0] * num_machines
    job_last_processed = [0] * num_jobs
    for i in sol:
        job = data[i]
        operation = job.pop(0)
        if machine_free_at[operation[0]] >= job_last_processed[i]:
            schedule[operation[0]].append((i, machine_free_at[operation[0]], operation[1]))
            machine_free_at[operation[0]] += operation[1]
            job_last_processed[i] = machine_free_at[operation[0]]
        else:
            schedule[operation[0]].append((i, job_last_processed[i], operation[1]))
            job_last_processed[i] += operation[1]
            machine_free_at[operation[0]] = job_last_processed[i]            
    return schedule

class JSSP_Individual(Individual):
    def __init__(self, solution):
        super().__init__(solution)
    
    def fitness(self, jssp_data, num_machines, num_jobs):
        sch = convert_solution_to_schedule(self.solution, copy.deepcopy(jssp_data), num_machines, num_jobs)
        cmax = []
        for i in sch:
            cmax.append(i[num_jobs-1][1] + i[num_jobs-1][2])
        return max(cmax)
    
class JSSP_Population(Population):
    def __init__(self, individuals):
        super().__init__(individuals)

    def fitness_scores(self, jssp_data, num_jobs, num_machines):
        return [individual.fitness(jssp_data, num_machines, num_jobs) for individual in self.individuals]

    def crossover(self, parent1, parent2, num_jobs, num_machines):
        # Child 1
        point1, point2 = sorted(random.sample(range(1, num_machines), 2))
        
        offspring1 = parent2.solution[:point1*num_jobs] + parent1.solution[point1*num_jobs:point2*num_jobs] + parent2.solution[point2*num_jobs:]

        # Perform crossover for child 2 (offspring2)
        offspring2 = parent1.solution[:point1*num_jobs] + parent2.solution[point1*num_jobs:point2*num_jobs] + parent1.solution[point2*num_jobs:]

        return JSSP_Individual(offspring1), JSSP_Individual(offspring2)

    def mutate(self, solution, num_jobs, num_machines):
        mutated_solution = solution.solution[:]
        random_index_1 = random.randint(0, num_machines-1)
        random_index_2 = random.randint(0, num_machines-1)
        while random_index_1 == random_index_2:
            random_index_2 = random.randint(0, num_machines-1)

        if random_index_1 > random_index_2:
            deleted_elements = mutated_solution[random_index_1*num_jobs:random_index_1*num_jobs+num_jobs]
            del mutated_solution[random_index_1*num_jobs:random_index_1*num_jobs+num_jobs]
            for i in range((random_index_2)*num_jobs, (random_index_2)*num_jobs+num_jobs):
                mutated_solution.insert(i, deleted_elements.pop(0))
        else:
            deleted_elements = mutated_solution[random_index_2*num_jobs:random_index_2*num_jobs+num_jobs]
            del mutated_solution[random_index_2*num_jobs:random_index_2*num_jobs+num_jobs]
            for i in range((random_index_1)*num_jobs, (random_index_1)*num_jobs+num_jobs):
                mutated_solution.insert(i, deleted_elements.pop(0))
        return JSSP_Individual(mutated_solution)

class JSSP_EA(EvolutionaryAlgorithm):
    def __init__(self, population_size, generations, mutation_rate, offsprings, num_jobs, num_machines):
        super().__init__(population_size, generations, mutation_rate, offsprings)
        self.num_jobs = num_jobs
        self.num_machines = num_machines

    def initialize_population(self):
        individuals = []
        for _ in range(self.population_size):
            random_solution = []
            for _ in range(self.num_machines):
                random_solution += random.sample(range(self.num_jobs), self.num_jobs)
            individual = JSSP_Individual(random_solution)
            individuals.append(individual)
        return JSSP_Population(individuals)

    def run(self, jssp_data, pop):
        best_fitness_values = []
        avg_fitness_values = []

        for generation in range(self.generations):
            fitness_scores = pop.fitness_scores(jssp_data, self.num_jobs, self.num_machines)

            # Create offspring through crossover and mutation
            offspring = []
            for _ in range(self.offsprings // 2):
                parent1 = pop.individuals[self.rank_based_selection_min(fitness_scores)]
                parent2 = pop.individuals[self.rank_based_selection_min(fitness_scores)]
                child1, child2 = pop.crossover(parent1, parent2, self.num_jobs, self.num_machines)
                random_number_1 = random.random()
                random_number_2 = random.random()
                if random_number_1 > self.mutation_rate:
                    offspring.append(child1)
                else:
                    child1 = pop.mutate(child1, self.num_jobs, self.num_machines)
                    offspring.append(child1)
                if random_number_2 > self.mutation_rate:
                    offspring.append(child2)
                else:
                    child2 = pop.mutate(child2, self.num_jobs, self.num_machines)
                    offspring.append(child2)

            for i in offspring:
                pop.individuals.append(i)

            fitness_scores = pop.fitness_scores(jssp_data, self.num_jobs, self.num_machines)

            temp_population = []
            for _ in range(self.population_size):
                x = self.truncation_selection_min(fitness_scores)
                y = pop.individuals[x]
                pop.individuals.pop(x)
                fitness_scores.pop(x)
                temp_population.append(y)
            pop.individuals = temp_population
            fitness_scores = pop.fitness_scores(jssp_data, self.num_jobs, self.num_machines)

            best_solution = min(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)

            best_fitness_values.append(best_solution)
            avg_fitness_values.append(average_fitness)

            print("Generation", generation+1, ": Best:", best_solution, "Average:", average_fitness)

        best_solution = min(fitness_scores)
        return pop, pop.individuals[fitness_scores.index(best_solution)], best_solution, best_fitness_values, avg_fitness_values
