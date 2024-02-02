import random
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Define parameters
POPULATION_SIZE = 500
GENERATIONS = 50
MUTATION_RATE = 0.25
OFFSPRINGS = 600


def convert_solution_to_schedule(sol, data, num_machines, num_jobs):
    schedule = [] 
    for i in range(num_machines): #num_machines
        schedule.append([])
    machine_free_at = [0] * num_machines #num_machines
    job_last_processed = [0] * num_jobs #num_jobs
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

# Define fitness function 
def fitness_function(solution, jssp_data, num_machines, num_jobs):
    sch = convert_solution_to_schedule(solution, copy.deepcopy(jssp_data), num_machines, num_jobs)
    cmax = []
    for i in sch:
        cmax.append(i[num_jobs-1][1]+jssp_data[i[num_jobs-1][0]][num_machines-1][1])    
    return max(cmax)

# Initialize population
def initialize_population(num_machines, num_jobs):
    popuation = []
    for i in range(POPULATION_SIZE):
        random_solution = []
        for i in range(num_machines): #num_machines
            random_solution += random.sample(range(num_jobs), num_jobs)
        popuation.append(random_solution)
    
    # print(popuation)
    return popuation

def crossover(parent1, parent2, num_jobs, num_machines):
    # Child 1
    point1, point2 = sorted(random.sample(range(1, num_machines), 2))
    
    offspring1 = parent2[:point1*num_jobs] + parent1[point1*num_jobs:point2*num_jobs] + parent2[point2*num_jobs:]

    # Perform crossover for child 2 (offspring2)
    offspring2 = parent1[:point1*num_jobs] + parent2[point1*num_jobs:point2*num_jobs] + parent1[point2*num_jobs:]

    return offspring1, offspring2

# Perform mutation
def mutate(solution, num_jobs, num_machines):
    mutated_solution = solution[:]
    random_index_1 = random.randint(0, num_machines-1)
    random_index_2 = random.randint(0, num_machines-1)
    while random_index_1 == random_index_2:
        random_index_2 = random.randint(0, num_machines-1)
        
    if random_index_1 > random_index_2:
        deleted_elements = mutated_solution[random_index_1*num_jobs:random_index_1*num_jobs+num_jobs]
        # print(deleted_elements)
        del mutated_solution[random_index_1*num_jobs:random_index_1*num_jobs+num_jobs]
        for i in range((random_index_2)*num_jobs, (random_index_2)*num_jobs+num_jobs):
            mutated_solution.insert(i, deleted_elements.pop(0))
    else:
        deleted_elements = mutated_solution[random_index_2*num_jobs:random_index_2*num_jobs+num_jobs]
        # print(deleted_elements)
        del mutated_solution[random_index_2*num_jobs:random_index_2*num_jobs+num_jobs]
        for i in range((random_index_1)*num_jobs, (random_index_1)*num_jobs+num_jobs):
            mutated_solution.insert(i, deleted_elements.pop(0))
        
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
def evolutionary_algorithm(num_machines, num_jobs, jssp_data):
    population = initialize_population(num_machines, num_jobs)

    for generation in range(GENERATIONS):
        # Evaluate population
        fitness_scores = [fitness_function(solution, jssp_data, num_machines, num_jobs) for solution in population]

        # Create offspring through crossover and mutation
        offspring = []
        for i in range(OFFSPRINGS//2):
            parent1 = population[fitness_proportional_selection(population,fitness_scores)]
            parent2 = population[fitness_proportional_selection(population,fitness_scores)]
            child1, child2 = crossover(parent1, parent2, num_jobs, num_machines)
            random_number_1 = random.random()
            random_number_2 = random.random()
            if random_number_1 > MUTATION_RATE:
                offspring.append(child1)
            else:
                child1 = mutate(child1,num_jobs, num_machines)
                offspring.append(child1)
            if random_number_2 > MUTATION_RATE:
                offspring.append(child2)
            else:
                child2 = mutate(child2, num_jobs, num_machines)
                offspring.append(child2)
            
        for i in offspring:
            population.append(i)
            
        fitness_scores = [fitness_function(solution, jssp_data, num_machines, num_jobs) for solution in population]
        temp_population = []
        for i in range(POPULATION_SIZE):
            x = truncation_selection_min(fitness_scores)
            y = population[x]
            population.pop(x)
            fitness_scores.pop(x)
            temp_population.append(y)
        population = temp_population
            
        best_solution = min(fitness_scores)
        average_fitness = average(fitness_scores)
        print("Generation", generation, ": Best:",best_solution, "Average:", average_fitness)
            
    best_solution = min(fitness_scores)
    average_fitness = average(fitness_scores)
    return population[fitness_scores.index(best_solution)], best_solution, average_fitness

def read_jssp_data(file_path):
    # file_path = 'jssp_2.txt'  # Replace with the actual file path
    with open(file_path, 'r') as file:
        datafile = file.read()

    # Parse input data
    lines = datafile.strip().split('\n')
    num_jobs, num_machines = map(int, lines[0].split())  # Line 0 contains the numbers of jobs and machines
    jssp_data = []
    for i in range(1,num_jobs+1):
        line = lines[i].split()
        x = []
        for j in range(0, num_machines*2, 2):
            tup = (int(line[j]), int(line[j+1]))
            # print(tup)
            x.append(tup)
        jssp_data.append(x)
    return num_jobs, num_machines, jssp_data

def average(lst):
    if not lst:
        return 0  # Handle the case when the list is empty
    return sum(lst) / len(lst)

def plot_chart(solution, jssp_data, num_machines, num_jobs):
    sch = convert_solution_to_schedule(solution, copy.deepcopy(jssp_data), num_machines, num_jobs)
    # Define a list of distinct colors for each job
    colors = plt.cm.Set3.colors

    # Your existing code to generate the data
    machines = []
    for i in range(num_machines-1, -1, -1):  # num_machines
        for j in range(num_jobs):  # num_jobs
            lst = []
            lst.append(["M" + str(i)])
            lst.append(sch[i][j][1])
            lst.append(sch[i][j][2])
            lst.append(sch[i][j][0])
            machines.append(lst)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot bars for each task with a unique color for each job
    for idx, (machine, start, duration, label) in enumerate(machines):
        color = colors[label % len(colors)]  # Cycle through colors if there are more jobs than colors
        ax.barh(machine, duration, left=start, color=color, edgecolor='blue')
        ax.text(start + duration / 2, machine, label, ha='center', va='center')

    # Add labels and formatting
    ax.set_xlabel('Time (mins)')
    ax.set_ylabel('Machine')
    ax.set_yticks(np.unique([machine for machine, _, _, _ in machines]))  # Set y-ticks for machines
    ax.invert_yaxis()  # Invert y-axis to match image orientation (optional)
    plt.grid(True)

    # Adjust layout for clarity
    plt.tight_layout()

    # Show the chart
    plt.show()


# Usage
filename = "jssp_3.txt"
num_jobs, num_machines, jssp_data = read_jssp_data(filename)
# pop = initialize_population(num_machines, num_jobs)

best_solution, best_fitness, avgFitness = evolutionary_algorithm(num_machines, num_jobs, jssp_data)
plot_chart(best_solution, jssp_data, num_machines, num_jobs)
print("Best fitness:", best_fitness, "Average fitness:", avgFitness)


