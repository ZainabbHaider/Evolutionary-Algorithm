import random
import math
import numpy as np

# Define parameters
POPULATION_SIZE = 40
GENERATIONS = 50
MUTATION_RATE = 0.2
OFFSPRINGS = 30

# Define fitness function 
def fitness_function(solution):
    fitness = 0
    for i in range(len(solution)-1):
        x1, y1 = tsp_data[solution[i]]
        x2, y2 = tsp_data[solution[i+1]]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        fitness+=distance
    
    return fitness

# Initialize population
def initialize_population(num_machines, num_jobs):
    m,n=num_machines, num_jobs
    
    matrix = np.zeros((m, n), dtype=int)
    
    # Generate unique permutations for rows
    for i in range(m):
        row_permutation = np.random.permutation(np.arange(1, n+1))
        matrix[i] = row_permutation
    return matrix

def crossover(parent1, parent2):
    # Child 1
    point1, point2 = sorted(random.sample(range(1, len(parent1)), 2))
    child1 = []
    child1_middle = parent2[point1:point2]  # Middle part from parent2
    remaining_num = []
    for i in range(point2, len(parent1)):
        if parent1[i] not in child1_middle:
            remaining_num.append(parent1[i])
    for i in range(point2):
        if parent1[i] not in child1_middle:
            remaining_num.append(parent1[i])
    for i in range(point2, len(parent1)):
        child1_middle.append(remaining_num.pop(0))
    
    for i in range(point1):
        child1.append(remaining_num.pop(0))
    child1 = child1 + child1_middle 
    
    # Child 2
    child2 = []
    child2_middle = parent1[point1:point2]  # Middle part from parent1
    remaining_num2 = []
    for i in range(point2, len(parent2)):
        if parent2[i] not in child2_middle:
            remaining_num2.append(parent2[i])
    for i in range(point2):
        if parent2[i] not in child2_middle:
            remaining_num2.append(parent2[i])
    for i in range(point2, len(parent2)):
        child2_middle.append(remaining_num2.pop(0))
    for i in range(point1):
        child2.append(remaining_num2.pop(0))
    child2 = child2 + child2_middle    
    
    return child1, child2

# Perform mutation
def mutate(solution):
    mutated_solution = solution[:]
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
        
    return mutated_solution

def random_selection(size):
    random_number = random.randint(0, size-1)
    return random_number

def truncation_selection_max(fitness_scores):
    max_value = max(fitness_scores)
    max_index = fitness_scores.index(max_value)
    return max_index

def truncation_selection_min(fitness_scores):
    min_index = min(fitness_scores)
    min_index = fitness_scores.index(min_index)
    return min_index

def fitness_proportional_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_index = random.choices(range(len(population)), weights=probabilities)
    return selected_index[0]

def rank_based_selection(population, fitness_scores):
    sorted_list = sorted(fitness_scores, reverse=True)
    indexed_list = list(enumerate(sorted_list, start=1))
    weight_mapping = {rank: fitness for rank, fitness in indexed_list}
    ranks = [index for index, _ in indexed_list]
    total_rank_sum = sum(ranks)
    probabilities = [rank / total_rank_sum for rank in ranks]
    selected_index = random.choices(ranks, weights = probabilities)
    return fitness_scores.index(weight_mapping[selected_index[0]])

def binary_tournament_selection(fitness_scores):
    r1 = random_selection(len(fitness_scores))
    r2 = random_selection(len(fitness_scores))
    while r1 == r2:
        r2 = random_selection(len(fitness_scores))
    if fitness_scores[r1] > fitness_scores[r2]:
        return r1
    else:
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
            parent1 = population[fitness_proportional_selection(population,fitness_scores)]
            parent2 = population[fitness_proportional_selection(population,fitness_scores)]
            random_number = random.random()
            if random_number>MUTATION_RATE:
                child1, child2 = crossover(parent1, parent2)
                offspring.append(child1)
                offspring.append(child2)
            else:
                child1 = mutate(parent1)
                child2 = mutate(parent2)
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
        best_solution = min(fitness_scores)
        average_fitness = average(fitness_scores)
        print("Generation", generation, ": Best:",best_solution, "Average:", average_fitness)
            
    best_solution = min(fitness_scores)
    average_fitness = average(fitness_scores)
    return population[fitness_scores.index(best_solution)], best_solution, average_fitness

def read_tsp_data(filename):
    # Read data from a text file
    file_path = 'data.txt'  # Replace with the actual file path
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the data into lines
    lines = data.split('\n')

    # Get the number of jobs and machines
    num_jobs, num_machines = map(int, lines[0].split())

    # Extract job processing times
    job_data = lines[1:]

    # Initialize an empty list to store job schedules
    job_schedules = []

    # Parse each job data and create a dictionary for each job
    for i in range(num_jobs):
        job_schedule = {}
        for j in range(num_machines):
            machine, time = map(int, job_data[i].split()[2*j:2*j+2])
            job_schedule[machine] = time
        job_schedules.append(job_schedule)

def average(lst):
    if not lst:
        return 0  # Handle the case when the list is empty
    return sum(lst) / len(lst)

# Usage
filename = "data.txt"
tsp_data = read_tsp_data(filename)
# print(tsp_data)
# pop = initialize_population()
# print(pop)

best_solution, best_fitness, avgFitness = evolutionary_algorithm()
# print(best_solution)
print("Best fitness:", best_fitness, "Average fitness:", avgFitness)
# print(fitness_function(pop[0]))


# p1 = [1.4, 5.6, 2.3, 7.2, 9.3 ,3.5, 6.5, 4.6, 0.7, 1.9]
# p2 = [9,3,7,8,2,6,5,1,4,0]
# print(rank_based_selection(p2,p1))
# crossover(p1,p2)

