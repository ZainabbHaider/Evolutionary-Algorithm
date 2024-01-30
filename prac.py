import random

# Read data from a text file
file_path = 'data.txt'  # Replace with the actual file path
with open(file_path, 'r') as file:
    data = file.read()

# Parse input data
lines = data.strip().split('\n')
num_jobs, num_machines = map(int, lines[0].split())  # Line 0 contains the numbers of jobs and machines
job_sequences = [list(map(int, line.split())) for line in lines[2:]]  # Start from line 3 for job sequences

# Create a 2D matrix to represent the sequence of jobs on each machine
machine_job_matrix = [[job_sequences[job_idx][machine_idx] for job_idx in range(num_jobs)] for machine_idx in range(num_machines)]

# Randomize the order of jobs within each row while maintaining the original order of jobs for each machine
for row in machine_job_matrix:
    original_order = row[:]
    random.shuffle(row)
    row.sort(key=lambda x: original_order.index(x))

# Print the randomized matrix
for row in machine_job_matrix:
    print(row)
