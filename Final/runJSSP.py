import random
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from JSSP_Implementation import *

# Define parameters
POPULATION_SIZE = 50
GENERATIONS = 50
MUTATION_RATE = 0.25
OFFSPRINGS = 60

def read_jssp_data(file_path):
    with open(file_path, 'r') as file:
        datafile = file.read()

    lines = datafile.strip().split('\n')
    num_jobs, num_machines = map(int, lines[0].split())
    jssp_data = []
    for i in range(1, num_jobs + 1):
        line = lines[i].split()
        x = []
        for j in range(0, num_machines * 2, 2):
            tup = (int(line[j]), int(line[j + 1]))
            x.append(tup)
        jssp_data.append(x)
    return num_jobs, num_machines, jssp_data

def plot_chart(solution, jssp_data, num_machines, num_jobs, cmax_time):
    sch = convert_solution_to_schedule(solution, copy.deepcopy(jssp_data), num_machines, num_jobs)
    colors = plt.cm.Set3.colors

    machines = []
    for i in range(num_machines - 1, -1, -1):
        for j in range(num_jobs):
            lst = []
            lst.append(["M" + str(i)])
            lst.append(sch[i][j][1])
            lst.append(sch[i][j][2])
            lst.append(sch[i][j][0])
            machines.append(lst)

    fig, ax = plt.subplots()

    for idx, (machine, start, duration, label) in enumerate(machines):
        color = colors[label % len(colors)]
        ax.barh(machine, duration, left=start, color=color, edgecolor='blue')
        ax.text(start + duration / 2, machine, label, ha='center', va='center')

    ax.axvline(cmax_time, color='lightblue', linestyle='--', linewidth=2, label='Cmax', zorder=10)

    ax.set_xlabel('Time (mins)')
    ax.set_ylabel('Machine')
    ax.set_yticks(np.unique([machine for machine, _, _, _ in machines]))
    ax.invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Usage
filename = "jssp_3.txt"
num_jobs, num_machines, jssp_data = read_jssp_data(filename)

# Initialize EA
ea = JSSP_EA(population_size=POPULATION_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE, offsprings=OFFSPRINGS, num_jobs=num_jobs, num_machines=num_machines)

# Initialize and run evolutionary algorithm
avg_BSF = [0 for _ in range(GENERATIONS)]
avg_ASF = [0 for _ in range(GENERATIONS)]
best_solutions = []

for iteration in range(1):
    # Initialize a new random population for each iteration
    population = ea.initialize_population()
    best_fitness_values = []
    average_fitness_values = []

    population, best_individual, best_fit, best_fitness_values, average_fitness_values = ea.run(jssp_data, population)
    best_solutions.append(best_individual)
    avg_BSF = [x + y for x, y in zip(avg_BSF, best_fitness_values)]
    avg_ASF = [x + y for x, y in zip(avg_ASF, average_fitness_values)]
    print(f"Iteration: {iteration + 1}, Best Fitness: {best_fit}")

    # Plot the chart for the best solution
    plot_chart(best_individual.solution, jssp_data, num_machines, num_jobs, best_fit)

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