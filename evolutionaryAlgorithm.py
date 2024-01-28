import random
import math

# Define parameters
POPULATION_SIZE = 30
GENERATIONS = 50
MUTATION_RATE = 0.5
OFFSPRINGS = 10

# Define fitness function (example)
def fitness_function(solution):
    # Example: Fitness is the sum of the solution's elements
    fitness = 0
    for i in range(len(solution)-1):
        x1, y1 = tsp_data[solution[i]]
        x2, y2 = tsp_data[solution[i+1]]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        fitness+=distance
    
    return fitness

# Initialize population
def initialize_population():
    popuation = []
    for i in range(POPULATION_SIZE):
        all_values = list(tsp_data.keys())
        # print(all_values)
        # Randomly sample 'size' number of unique values from the list
        random.shuffle(all_values)
        # print(all_values)
        popuation.append(all_values)
    
    return popuation

# Perform crossover
def crossover(parent1, parent2):
    point = random.randint(0, len(parent1) - 1)
    child1 = parent1[point:]  # Copy the part after the cut point from parent1 to child1
    remaining_numbers = [num for num in parent2 if num not in child1]  # Numbers not in child1

    # Copy remaining numbers using the order of parent2
    for num in parent2:
        if len(child1) == len(parent1):  # If child1 is complete, break
            break
        if num not in child1:  # If the number is not in child1, add it
            child1.append(num)

    # Create child2 by swapping parent1 and parent2
    child2 = [num for num in parent1 if num not in child1]  # Numbers not in child1
    for num in parent2:
        if len(child2) == len(parent1):  # If child2 is complete, break
            break
        if num not in child2:  # If the number is not in child2, add it
            child2.append(num)

    return child1, child2

# Perform mutation
def mutate(solution):
    mutated_solution = solution[:]
    random_index_1 = random.randint(0, len(mutated_solution)-1)
    random_index_2 = random.randint(0, len(mutated_solution)-1)
    while random_index_1 == random_index_2:
        random_index_2 = random.randint(0, len(mutated_solution)-1)
    removed_item = mutated_solution.pop(random_index_1)
    mutated_solution.insert(random_index_2, removed_item)
    return mutated_solution

def random_selection(size):
    random_number = random.randint(0, size-1)
    return random_number

def truncation_selection_max(fitness_scores):
    max_value = max(fitness_scores)
    max_index = fitness_scores.index(max_value)
    return max_index

def truncation_selection_min(fitness_scores):
    max_value = min(fitness_scores)
    max_index = fitness_scores.index(max_value)
    return max_index

def fitness_proportional_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_index = random.choices(population, weights=probabilities)
    return population.index(selected_index[0])

def rank_based_selection(population, fitness_scores):
    indexed_list = list(enumerate(fitness_scores, start=1))
    sorted_list = sorted(indexed_list, key=lambda x: x[1])
    weight_mapping = {index: weight for weight, (index, _) in enumerate(sorted_list, start=1)}
    ranks = [weight_mapping[index] for index, _ in indexed_list]
    total_rank_sum = sum(ranks)
    probabilities = [rank / total_rank_sum for rank in ranks]
    selected_index = random.choices(population, weights = probabilities)
    return population.index(selected_index[0])

def binary_tournament_selection(fitness_scores):
    r1 = random_selection(len(fitness_scores))
    r2 = random_selection(len(fitness_scores))
    while r1 == r2:
        r2 = random_selection(len(fitness_scores))
    if fitness_scores[r1] > fitness_scores[r2]:
        return r1
    else
        return r2

# Evolutionary algorithm
def evolutionary_algorithm():
    population = initialize_population()

    for generation in range(GENERATIONS):
        # Evaluate population
        fitness_scores = [fitness_function(solution) for solution in population]

        # Create offspring through crossover and mutation
        offspring = []
        for i in range(OFFSPRINGS//2):
            parent1 = population[fitness_proportional_selection(population, fitness_scores)]
            parent2 = population[fitness_proportional_selection(population, fitness_scores)]
            child1, child2 = crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
            
        for i in offspring:
            population.append(i)
            
        fitness_scores = [fitness_function(solution) for solution in population]
        for i in range(OFFSPRINGS):
            i = truncation_selection_max(fitness_scores)
            # print(i)
            population.pop(i)
            fitness_scores.pop(i)
        # print(len(population))

    # Output the best solution found
    best_solution = max(population, key=fitness_function)
    return best_solution, fitness_function(best_solution)

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

# Usage
filename = "qa194.tsp"
tsp_data = read_tsp_data(filename)
# print(tsp_data)
# pop = initialize_population()
# print(pop)

best_solution, best_fitness = evolutionary_algorithm()
print(best_solution)
print(best_fitness)
# print(fitness_function(pop[0]))




