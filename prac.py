import random
import copy
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


# Read data from a text file
file_path = 'jssp_2.txt'  # Replace with the actual file path
with open(file_path, 'r') as file:
    datafile = file.read()

# Parse input data
lines = datafile.strip().split('\n')
num_jobs, num_machines = map(int, lines[0].split())  # Line 0 contains the numbers of jobs and machines
jsp_data = []
for i in range(1,num_jobs+1):
    line = lines[i].split()
    x = []
    for j in range(0, num_machines*2, 2):
        tup = (int(line[j]), int(line[j+1]))
        # print(tup)
        x.append(tup)
    jsp_data.append(x)

# print(jsp_data)
# print(jsp_data[9])

def random_permutation(num_jobs):
    return random.sample(range(num_jobs), num_jobs)

random_solution = []
for i in range(num_machines): #num_machines
    random_solution += random_permutation(num_jobs) #num_jobs

print(random_solution)
# print(len(random_solution))

# convert random solution to a schedule here
def convert_to_schedule(sol, data):
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
    return schedule, job_last_processed

sol = [0, 2, 3, 1, 3, 1, 2, 0, 1, 0, 3, 2]
data2 = [[(0, 3), (1, 2), (2, 2)], [(1, 2), (2, 1), (0, 4)], [(0, 4), (2, 3), (1, 1)], [(2, 5), (1, 2), (0, 3)]]
sch, jlp = convert_to_schedule(random_solution, copy.deepcopy(jsp_data))

cmax = []
for i in sch:
    cmax.append(i[num_jobs-1][1]+jsp_data[i[num_jobs-1][0]][num_machines-1][1])
    # cmax.append(i[4-1][1]+jsp_data[i[4-1][0]][3-1][1])

print(max(cmax))

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

def mutate(solution, num_jobs, num_machines):
    mutated_solution = solution[:]
    random_index_1 = random.randint(0, num_machines-1)
    random_index_2 = random.randint(0, num_machines-1)
    while random_index_1 == random_index_2:
        random_index_2 = random.randint(0, num_machines-1)
        
    if random_index_1 > random_index_2:
        deleted_elements = mutated_solution[random_index_1*num_jobs:random_index_1*num_jobs+num_jobs]
        print(deleted_elements)
        del mutated_solution[random_index_1*num_jobs:random_index_1*num_jobs+num_jobs]
        for i in range((random_index_2)*num_jobs, (random_index_2)*num_jobs+num_jobs):
            mutated_solution.insert(i, deleted_elements.pop(0))
    else:
        deleted_elements = mutated_solution[random_index_2*num_jobs:random_index_2*num_jobs+num_jobs]
        print(deleted_elements)
        del mutated_solution[random_index_2*num_jobs:random_index_2*num_jobs+num_jobs]
        for i in range((random_index_1)*num_jobs, (random_index_1)*num_jobs+num_jobs):
            mutated_solution.insert(i, deleted_elements.pop(0))
        
    return mutated_solution

def crossover(parent1, parent2, point1, point2, num_jobs):
    # Perform crossover for child 1 (offspring1)
    offspring1 = parent2[:point1*num_jobs] + parent1[point1*num_jobs:point2*num_jobs] + parent2[point2*num_jobs:]

    # Perform crossover for child 2 (offspring2)
    offspring2 = parent1[:point1*num_jobs] + parent2[point1*num_jobs:point2*num_jobs] + parent1[point2*num_jobs:]

    return offspring1, offspring2

# Example usage:
parent1 = [1, 2, 3, 3, 2, 1, 2, 3, 1, 2, 1, 3]
parent2 = [2, 3, 1, 3, 1, 2, 3, 2, 1, 1, 3, 2]
point1 = 1
point2 = 2
print(mutate(parent1, 3, 4))
# offspring1, offspring2 = crossover(parent1, parent2, point1, point2, 3)
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Offspring 1:", offspring1)
# print("Offspring 2:", offspring2)
