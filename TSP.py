import random
import numpy as np
from itertools import permutations
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
INT_MAX = 999999
CITIES = 10
TESTS = 50
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
    
    def one_point_crossover(self, partner):
        point =  random.randint(0, len(self.chromosome) - 1)
        child1 = self.chromosome[:point] + partner.chromosome[point:]
        child2 = partner.chromosome[:point] + self.chromosome[point:]
        return Individual(child1), Individual(child2) #maybe stuck local minima
    
    def uniform_crossover(self, partner):
        child1 = list(self.chromosome)
        child2 = list(partner.chromosome)
        for i in range(len(self.chromosome)):
            if bool(random.getrandbits(1)):
                child1[i], child2[i] = child2[i], child1[i]
        return Individual(child1), Individual(child2)
    
    def cycle_crossover(self, partner):
        child1 = [None] * len(self.chromosome)
        child2 = [None] * len(self.chromosome)

        visited = set()
        start_index = random.randint(0, len(self.chromosome) - 1)
        current_index = start_index

        while current_index not in visited:
            visited.add(current_index)
            child1[current_index] = self.chromosome[current_index]
            child2[current_index] = partner.chromosome[current_index]

            next_index = partner.chromosome.index(self.chromosome[current_index])
            current_index = next_index

        for i in range(len(self.chromosome)):
            if i not in visited:
                child1[i] = self.chromosome[i]
                child2[i] = partner.chromosome[i]

        return Individual(child1), Individual(child2)
    
    def PMX_crossover(self, partner):
        start = random.randint(0, len(self.chromosome) - 1)
        end = random.randint(start, len(self.chromosome) - 1)

        child1 = [None] * len(self.chromosome)
        child2 = [None] * len(self.chromosome)
        
        child1[start:end+1] = partner.chromosome[start:end+1]
        child2[start:end+1] = self.chromosome[start:end+1]

        mapping1 = {}
        mapping2 = {}

        for i in range(start, end + 1):
            mapping1[self.chromosome[i]] = partner.chromosome[i]
            mapping2[partner.chromosome[i]] = self.chromosome[i]

        for i in range(len(self.chromosome)):
            if child1[i] is None:
                child1[i] = mapping1.get(self.chromosome[i], self.chromosome[i])
            if child2[i] is None:
                child2[i] = mapping2.get(partner.chromosome[i], partner.chromosome[i])

        return Individual(child1), Individual(child2) # watch
    #order crossover
    def order_crossover(self, partner):
        start = random.randint(0, len(self.chromosome) - 1)
        end = random.randint(start, len(self.chromosome) - 1) 

        child1 = [None] * len(self.chromosome)
        child2 = [None] * len(self.chromosome)
        
        child1[start:end+1] = self.chromosome[start:end+1]
        child2[start:end+1] = partner.chromosome[start:end+1]

        i, j, k, l = (end + 1) % len(self.chromosome), (end + 1) % len(self.chromosome), (end + 1) % len(self.chromosome), (end + 1) % len(self.chromosome)
        while k != end and l != end:
            while self.chromosome[i] in child2 and i != end:
                i = (i + 1) % len(self.chromosome)
            while partner.chromosome[j] in child1 and j != end:
                j = (j + 1) % len(self.chromosome)

            child1[k] = partner.chromosome[j]
            child2[l] = self.chromosome[i]
            if k != end:
                k = (k + 1) % len(self.chromosome)
            if l != end:
                l = (l + 1) % len(self.chromosome)

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
    fitness_scores = [individual.fitness for individual in population]
    total_fitness = sum(fitness_scores)
    roll = random.uniform(0, total_fitness)
    cumulative_probability  = 0
    for chromosome in population:
        cumulative_probability += chromosome.fitness
        if cumulative_probability >= roll:
            return chromosome
        
def rank_selection(population):
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
    for i in range(TESTS):
        random.seed(i)
        distance_matrix = create_graph(CITIES, 1, 10000)
        gene = genetic_algorithm()
        #print(distance_matrix)
        brute_force = brute_force_algorithm()
        if(gene == brute_force):
            correct_path+=1

    print(f"Percentage of found path is shortest: {(correct_path/TESTS) * 100}%")


#2.10.2024: code mutation/crossover
#Next: code selection
#Next_1: 1 more base case
#Next_2: Analysts