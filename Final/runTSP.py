import random
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from TSP_Implentation import *
 
POPULATION_SIZE = 100
GENERATIONS = 10000
MUTATION_RATE = 0.7
OFFSPRINGS = 120

def read_tsp_data(filename):
    tsp_data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

        # Find the index where NODE_COORD_SECTION starts
        coord_section_index = lines.index('NODE_COORD_SECTION\n')

        # Iterate over lines after NODE_COORD_SECTION and extract coordinates
        for line in lines[coord_section_index + 1:]:
            if line.strip() == 'EOF':
                break
            node, x, y = line.strip().split(' ')
            tsp_data[int(node)] = (float(x), float(y))

    return tsp_data

filename = "qa194.tsp"
tsp_data = read_tsp_data(filename)

ea = TSP_EA(population_size= POPULATION_SIZE, generations= GENERATIONS, mutation_rate= MUTATION_RATE, offsprings= OFFSPRINGS, replacement=0, elite=0)

# Initialize and run evolutionary algorithm
avg_BSF = [0 for _ in range(GENERATIONS)]
avg_ASF = [0 for _ in range(GENERATIONS)]
best_solutions = []

for iteration in range(1):
    # Initialize a new random population for each iteration
    population = ea.initialize_population(tsp_data)
    best_fitness_values = []
    average_fitness_values = []
    
    population, best_individual, best_fit, best_fitness_values, average_fitness_values = ea.run(tsp_data, population)
    best_solutions.append(best_individual)
    avg_BSF = [x + y for x, y in zip(avg_BSF, best_fitness_values)]
    avg_ASF = [x + y for x, y in zip(avg_ASF, average_fitness_values)]
    print(f"Iteration: {iteration + 1}, Best Fitness: {best_fit}")

# Calculate average fitness over iterations
avg_BSF = [x / 1 for x in avg_BSF]
avg_ASF = [x / 1 for x in avg_ASF]

# Plotting
generations = range(1, len(best_fitness_values) + 1)

plt.plot(generations, avg_BSF, label='Mean Best Fitness')
plt.plot(generations, avg_ASF, label='Mean Average Fitness')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('Mean Best and Average Fitness over Iterations')
plt.legend()
plt.show()

halloffame_filename = "HallofFame.txt"
with open(halloffame_filename, 'a') as halloffame_file:
    for i in best_solutions:
        fitness_score = i.fitness(tsp_data)
        if fitness_score <= 12000:
            halloffame_file.write(f"{i.solution}; {i.fitness(tsp_data)}\n")