import random
import math

class Individual:
    def __init__(self, solution):
        self.solution = solution

    def fitness(self, tsp_data):
        fitness = 0
        for i in range(len(self.solution) - 1):
            x1, y1 = tsp_data[self.solution[i]]
            x2, y2 = tsp_data[self.solution[i + 1]]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            fitness += distance
        return fitness

class Population:
    def __init__(self, individuals):
        self.individuals = individuals

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
        c1 = Individual(child1)
        
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
        c2 = Individual(child2)
        
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

        m = Individual(mutated_solution)
        return m

    def random_selection(self, size):
        random_number = random.randint(0, size-1)
        return random_number

    def truncation_selection_max(self, fitness_scores):
        max_value = max(fitness_scores)
        max_index = fitness_scores.index(max_value)
        return max_index

    def truncation_selection_min(self, fitness_scores):
        min_index = min(fitness_scores)
        min_index = fitness_scores.index(min_index)
        return min_index

    def fitness_proportional_selection(self, fitness_values):
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        selected_index = random.choices(range(len(self.individuals)), weights=probabilities)
        return selected_index[0]

    def rank_based_selection(self, fitness_scores):
        sorted_list = sorted(fitness_scores, reverse=True)
        indexed_list = list(enumerate(sorted_list, start=1))
        weight_mapping = {rank: fitness for rank, fitness in indexed_list}
        ranks = [index for index, _ in indexed_list]
        total_rank_sum = sum(ranks)
        probabilities = [rank / total_rank_sum for rank in ranks]
        selected_index = random.choices(ranks, weights = probabilities)
        return fitness_scores.index(weight_mapping[selected_index[0]])

    def binary_tournament_selection(self, fitness_scores):
        r1 = self.random_selection(len(fitness_scores))
        r2 = self.random_selection(len(fitness_scores))
        while r1 == r2:
            r2 = self.random_selection(len(fitness_scores))
        if fitness_scores[r1] > fitness_scores[r2]:
            return r1
        else:
            return r2

class EvolutionaryAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, offsprings):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.offsprings = offsprings

    def initialize_population(self, tsp_data):
        individuals = []
        for i in range(self.population_size):
            all_values = list(tsp_data.keys())
            random.shuffle(all_values)
            individual = Individual(all_values)
            individuals.append(individual)
        return Population(individuals)

    def run(self, tsp_data, pop):
        

        for generation in range(self.generations):
            fitness_scores = pop.fitness_scores(tsp_data)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(self.offsprings // 2):
                parent1 = pop.individuals[pop.rank_based_selection(fitness_scores)]
                parent2 = pop.individuals[pop.rank_based_selection(fitness_scores)]
                random_number = random.random()
                if random_number>self.mutation_rate:
                    child1, child2 = pop.crossover(parent1, parent2)
                    offspring.append(child1)
                    offspring.append(child2)
                else:
                    child1 = pop.mutate(parent1)
                    child2 = pop.mutate(parent2)
                    offspring.append(child1)
                    offspring.append(child2)
                    
            for i in offspring:
                pop.individuals.append(i)
            
            fitness_scores = pop.fitness_scores(tsp_data)

            for i in range(self.offsprings):
                i = pop.truncation_selection_max(fitness_scores)
                pop.individuals.pop(i)
                fitness_scores.pop(i)
            best_solution = min(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            print("Generation", generation, ": Best:",best_solution, "Average:", average_fitness)
                
        best_solution = min(fitness_scores)
        average_fitness = sum(fitness_scores) / len(fitness_scores)
        return pop, pop.individuals[fitness_scores.index(best_solution)], best_solution, average_fitness


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

# Example usage
filename = "qa194.tsp"
tsp_data = read_tsp_data(filename)

ea = EvolutionaryAlgorithm(population_size=40, generations=1000, mutation_rate=0.2, offsprings=30)
population = ea.initialize_population(tsp_data)
# Initialize and run evolutionary algorithm
for iteration in range(10):
    population, best_individual, best_fit, avg_fit = ea.run(tsp_data,population)
    # best_fitness = best_individual.fitness(tsp_data)
    print(f"Iteration: {iteration} Best Fitness: {best_fit}")
