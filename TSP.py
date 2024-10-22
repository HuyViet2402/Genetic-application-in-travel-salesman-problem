import random
import time
import numpy as np
from itertools import permutations
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
INT_MAX = 999999
CITIES = 10
TESTS = 10
GENERATIONS = 50
# Distance matrix: distance_matrix[i][j] is the distance from city i to city j
def create_graph(cities, min_dis, max_dis):
    graph = [[0] * cities for _ in range(cities)]  # Create an empty 2D list
    for i in range(cities):
        for j in range(i + 1, cities):
            # Generate a random distance within the specified range
            distance = random.randint(min_dis, max_dis)
            graph[i][j] = distance
            graph[j][i] = distance  # For undirected graphs

    return graph

distance_matrix = create_graph(CITIES, 1, 10000)

class Individual(object):
    ''' Class representing individual in the population (a tour) '''

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()

    @classmethod
    def create_gnome(self):
        ''' Create a random chromosome (tour) by shuffling cities '''
        gnome = list(range(len(distance_matrix)))
        random.shuffle(gnome)
        return gnome
################################### Crossover ################################################
    def mate(self, partner):
        ''' Perform crossover and produce offspring '''
        child_chromosome = []

        # Select a subset of cities from parent 1
        start = random.randint(0, len(self.chromosome) - 1)
        end = random.randint(start, len(self.chromosome) - 1)
        child_chromosome[start:end] = self.chromosome[start:end]

        # Fill the rest from parent 2, maintaining order
        for city in partner.chromosome:
            if city not in child_chromosome:
                child_chromosome.append(city)

        return Individual(child_chromosome)
    
    # def one_point_crossover(self, partner):
    #     point =  random.randint(0, len(self.chromosome) - 1)
    #     child1 = self.chromosome[:point] + partner.chromosome[point:]
    #     child2 = partner.chromosome[:point] + self.chromosome[point:]
    #     return Individual(child1), Individual(child2)

    def one_point_crossover(self, partner):
        # Generate a random crossover point
        point = random.randint(0, len(self.chromosome) - 1)

        # Create child templates from the crossover point
        child1 = self.chromosome[:point]
        child2 = partner.chromosome[:point]

        for gene in partner.chromosome:
            if gene not in child1:
                child1.append(gene)

        for gene in self.chromosome:
            if gene not in child2:
                child2.append(gene)

        return Individual(child1), Individual(child2)

    # def uniform_crossover(self, partner):
    #     child1 = list(self.chromosome)
    #     child2 = list(partner.chromosome)
    #     for i in range(len(self.chromosome)):
    #         if bool(random.getrandbits(1)):
    #             child1[i], child2[i] = child2[i], child1[i]
    #     return Individual(child1), Individual(child2)

    def uniform_crossover(self, partner):
        # Initialize two empty chromosomes for the offspring
        child1 = [None] * len(self.chromosome)
        child2 = [None] * len(self.chromosome)

        # Track genes already added to avoid duplicates
        used_in_child1 = set()
        used_in_child2 = set()

        # Perform uniform crossover by iterating through each gene
        for i in range(len(self.chromosome)):
            if bool(random.getrandbits(1)):
                # Swap genes between parents for child1 and child2
                if partner.chromosome[i] not in used_in_child1:
                    child1[i] = partner.chromosome[i]
                    used_in_child1.add(partner.chromosome[i])
                if self.chromosome[i] not in used_in_child2:
                    child2[i] = self.chromosome[i]
                    used_in_child2.add(self.chromosome[i])
            else:
                # Keep the genes from the respective parents
                if self.chromosome[i] not in used_in_child1:
                    child1[i] = self.chromosome[i]
                    used_in_child1.add(self.chromosome[i])
                if partner.chromosome[i] not in used_in_child2:
                    child2[i] = partner.chromosome[i]
                    used_in_child2.add(partner.chromosome[i])
        # Fill in any missing genes to avoid None values
        for i in range(len(self.chromosome)):
            if child1[i] is None:
                for gene in self.chromosome:
                    if gene not in used_in_child1:
                        child1[i] = gene
                        used_in_child1.add(gene)
                        break
            if child2[i] is None:
                for gene in partner.chromosome:
                    if gene not in used_in_child2:
                        child2[i] = gene
                        used_in_child2.add(gene)
                        break
        return Individual(child1), Individual(child2)
    def cycle_crossover(self, partner):
        child1 = [None] * len(self.chromosome)
        child2 = [None] * len(self.chromosome)

        visited = set()
        start_index = 0
        current_index = start_index

        while current_index not in visited:
            visited.add(current_index)
            child1[current_index] = self.chromosome[current_index]
            child2[current_index] = partner.chromosome[current_index]

            next_index = partner.chromosome.index(self.chromosome[current_index])
            current_index = next_index

        for i in range(len(self.chromosome)):
            if i not in visited:
                child1[i] = partner.chromosome[i]
                child2[i] = self.chromosome[i]

        return Individual(child1), Individual(child2)
    
    # def PMX_crossover(self, partner):
    #     start = random.randint(0, len(self.chromosome) - 1)
    #     end = random.randint(start, len(self.chromosome) - 1)

    #     child1 = [None] * len(self.chromosome)
    #     child2 = [None] * len(self.chromosome)
        
    #     child1[start:end+1] = partner.chromosome[start:end+1]
    #     child2[start:end+1] = self.chromosome[start:end+1]

    #     mapping1 = {}
    #     mapping2 = {}

    #     for i in range(start, end + 1):
    #         mapping1[self.chromosome[i]] = partner.chromosome[i]
    #         mapping2[partner.chromosome[i]] = self.chromosome[i]

    #     for i in range(len(self.chromosome)):
    #         if child1[i] is None:
    #             child1[i] = mapping1.get(self.chromosome[i], self.chromosome[i])
    #         if child2[i] is None:
    #             child2[i] = mapping2.get(partner.chromosome[i], partner.chromosome[i])

    #     return Individual(child1), Individual(child2) # watch

    def PMX_crossover(self, partner):
        start = random.randint(0, len(self.chromosome) - 1)
        end = random.randint(start, len(self.chromosome) - 1)

        print("start:", start)
        print("end:", end)
        def PMX_one_offspring(p1, p2):
            offspring = np.zeros(len(p1), dtype=p1.dtype)

            # Copy the mapping section (middle) from parent1
            offspring[start:end + 1] = p1[start:end + 1]
            print(offspring)
            # copy the rest from parent2 (provided it's not already there
            for i in np.concatenate([np.arange(0, start), np.arange(end + 1, len(p1))]):
                candidate = p2[i]
                while candidate in p1[start:end + 1]:  # allows for several successive mappings
                    print(f"Candidate {candidate} not valid in position {i}")  # DEBUGONLY
                    candidate = p2[np.where(p1 == candidate)[0][0]]
                    print(candidate)
                offspring[i] = candidate
            return offspring

        child1 = PMX_one_offspring(np.array(self.chromosome), np.array(partner.chromosome))
        child2 = PMX_one_offspring(np.array(partner.chromosome), np.array(self.chromosome))

        return Individual(child1.tolist()), Individual(child2.tolist())
    
    def order_crossover(self, partner):
        start = random.randint(0, len(self.chromosome) - 2)
        end = random.randint(start+1, len(self.chromosome) - 1) 
        #push tomorrow
        child1 = [None] * len(self.chromosome)
        child2 = [None] * len(self.chromosome)
        
        child1[start:end+1] = self.chromosome[start:end+1]
        child2[start:end+1] = partner.chromosome[start:end+1]

        startIndex = (end + 1) % len(self.chromosome)
        j = (end + 1) % len(self.chromosome)
        k = (end + 1) % len(self.chromosome)
        for i in range(startIndex, len(self.chromosome)):
            if partner.chromosome[i] not in child1:
                child1[j] = partner.chromosome[i] 
                j = (j + 1) % len(self.chromosome)
            if self.chromosome[i] not in child2:
                child2[k] = self.chromosome[i] 
                k = (k + 1) % len(self.chromosome)

        for i in range(0, startIndex):
            if partner.chromosome[i] not in child1:
                child1[j] = partner.chromosome[i] 
                j = (j + 1) % len(self.chromosome)
            if self.chromosome[i] not in child2:
                child2[k] = self.chromosome[i] 
                k = (k + 1) % len(self.chromosome)
        return Individual(child1), Individual(child2)
################################### Crossover ################################################
################################### Mutation ################################################
    def swap_mutate(self):
        ''' Mutate the chromosome by swapping two cities '''
        idx1 = random.randint(0, len(self.chromosome) - 1)
        idx2 = random.randint(0, len(self.chromosome) - 1)
        self.chromosome[idx1], self.chromosome[idx2] = self.chromosome[idx2], self.chromosome[idx1]

    def inverse_mutate(self):
        idx1 = random.randint(0, len(self.chromosome) - 1)
        idx2 = random.randint(0, len(self.chromosome) - 1)

        substring = self.chromosome[idx1:idx2 + 1]
        reversed_substring = substring[::-1]
        self.chromosome = self.chromosome[:idx1] + reversed_substring + self.chromosome[idx2 + 1:]

    def scramble_mutate(self):
        idx1 = random.randint(0, len(self.chromosome) - 1)
        idx2 = random.randint(0, len(self.chromosome) - 1)

        substring = list(self.chromosome[idx1:idx2 + 1])
        random.shuffle(substring)
        self.chromosome = self.chromosome[:idx1] + substring + self.chromosome[idx2 + 1:]
################################### Mutation ################################################
    def calculate_fitness(self):
        ''' Calculate fitness as the total distance of the tour '''
        total_distance = 0
        for i in range(len(self.chromosome) - 1):
            city1 = self.chromosome[i]
            city2 = self.chromosome[i + 1]
            total_distance += distance_matrix[city1][city2]

        # Add the distance to return to the starting city
        #total_distance += distance_matrix[self.chromosome[-1]][self.chromosome[0]]
        return total_distance
    def check_valid_path(self):
        seen = set()
        for city in self.chromosome:
            if city in seen:
                return False
            seen.add(city)
        return True
####################################################################################################
################################### Selection ################################################
def random_selection(population):
    return random.choice(population)

def roulette_wheel(population):
    ''' Choosing parent based on their fitness
        The lower the fitness, the higher chance it get picked
        The chance is proportional to the fitness
        n is population size
        Time complexity: O(3n)
        Space complexity: O(2n)
    '''
    fitness_scores = [1/individual.fitness for individual in population]
    total_fitness = sum(fitness_scores)
    probability_individual = [fitness/total_fitness for fitness in fitness_scores]
    return np.random.choice(population, p=probability_individual)
        
def rank_selection(population):
    ''' Choosing parent based on their fitness
        Assign each individual rank based on their fitness
        The higher the fitness, the higher the rank, the higher the chance to get selected
        The chance of getting selected is not propotional to the fitness
        n is population size
        Time complexity: O(n log n) + O(2n)
        Space complexity: O(n)
    '''
    population = sorted(population, key=lambda x: x.fitness, reverse=True)
    rank_sum = POPULATION_SIZE * (POPULATION_SIZE + 1) / 2
    probability = [i/rank_sum for i in range(1, POPULATION_SIZE + 1)]
    roll = random.random()
    cumulative_probability  = 0
    for i in range(0, POPULATION_SIZE):
        cumulative_probability += probability[i]
        if cumulative_probability >= roll:
            return population[i]
    
def tournament_selection(population, k=10):
    ''' Select k individuals from parents
        Select the best individual from k
        n is population size
        Time complexity: O(1)
        Space complexity: O(1)
    '''
    start = random.randint(0, POPULATION_SIZE - k)
    end = start + k
    return min(population[start:end], key=lambda x: x.fitness)
################################### Selection ################################################
####################################################################################################
def genetic_algorithm():
    # Initialize population
    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]
    generation = 1
    count = GENERATIONS
    while count > 0:
        # Sort population by fitness (lower is better)
        population = sorted(population, key=lambda x: x.fitness)
        for i in range(len(population)):
            print(f"ID_tour: {i} | Tour: {population[i].chromosome} | Fitness: {population[i].fitness}")
        # Print the best solution in the population
        print(
            f"Generation: {generation} | Best Fitness: {population[0].fitness:.2f} | Tour: {population[0].chromosome}")

        # If the best fitness is small enough, stop the algorithm
        # if population[0].fitness < 14000:
        #     break

        # Create a new generation
        new_generation = []

        # Elitism: Carry over the top 10% of the population
        elite_size = int(0.1 * POPULATION_SIZE)
        new_generation.extend(population[:elite_size])

        # Crossover: Generate the rest of the population by mating
        #for _ in range(POPULATION_SIZE - elite_size):
        while (len(new_generation) < int(POPULATION_SIZE * 0.8)):
            parent1 = rank_selection(population) 
            parent2 = rank_selection(population) 
            child1, child2= parent1.order_crossover(parent2)
            if random.random() < 0.1:
                child1.swap_mutate()  # Mutate the child
            if random.random() < 0.1:
                child2.swap_mutate()  # Mutate the child
            if(child1.check_valid_path()):
                new_generation.append(child1)
            if(child2.check_valid_path()):
                new_generation.append(child2)

        while(len(new_generation) < POPULATION_SIZE):
            new_generation.append(Individual(Individual.create_gnome()))
        # Replace the old population with the new one
        population = new_generation
        generation += 1
        count -= 1

    return population[0].fitness


def brute_force_algorithm():
    #n: number of cites
    # Time complexity: O(n!)
    # Space complexity: O(1)
    min = INT_MAX
    min_path = []
    for path in (permutations(range(0, CITIES))):
        distance = 0
        for i in range(len(path) - 1):
            city1 = path[i]
            city2 = path[i + 1]
            distance += distance_matrix[city1][city2]
            if distance > min:
                break
        
        if min > distance:
            min = distance
            min_path = path

    print(f"Best path: {min_path} | Distance: {min}")
    return min
            
        

if __name__ == '__main__':
    correct_path = 0
    brute_force_time_performance = 0
    genetic_time_performance = 0
    for i in range(TESTS):
        random.seed(i)
        distance_matrix = create_graph(CITIES, 1, 10000)

        gene_start = time.perf_counter()
        gene = genetic_algorithm()
        gene_end = time.perf_counter()
        print(f"Genetic algorithm runtime: {gene_end - gene_start:0.5f} seconds")
        genetic_time_performance += (gene_end - gene_start)

        brute_force_start = time.perf_counter()
        brute_force = brute_force_algorithm()
        brute_force_end = time.perf_counter()
        print(f"Brute force algorithm runtime: {brute_force_end - brute_force_start:0.5f} seconds")
        brute_force_time_performance += (brute_force_end - brute_force_start)

        if(gene == brute_force):
            correct_path+=1

    print(f"Percentage of correct path found by genetic algorithm: {(correct_path/TESTS) * 100}%")
    print(f"Total brute force algorithm runtime: {brute_force_time_performance:0.5f} seconds")
    print(f"Total genetic algorithm algorithm runtime: {genetic_time_performance:0.5f} seconds")