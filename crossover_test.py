import random
import time
import numpy as np
from itertools import permutations
POPULATION_SIZE = 3
MUTATION_RATE = 0.1
INT_MAX = 999999
CITIES = 10
TESTS = 10
GENERATIONS = 50

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
        gnome = list(range(CITIES))
        random.shuffle(gnome)
        return gnome
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
        ''' Choose a random point in the chromosome
            Genetic information beyond that point will be swaped with each other
            n is number of cities
            Time complexity: O(1)
            Space complexity: O(1)
        '''
        point =  random.randint(0, len(self.chromosome) - 1)
        child1 = self.chromosome[:point] + partner.chromosome[point:]
        child2 = partner.chromosome[:point] + self.chromosome[point:]
        return Individual(child1), Individual(child2)
    
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

if __name__ == '__main__':
    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]
    route_a = Individual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    route_b = Individual([4, 5 ,2, 1, 8, 0, 7, 6, 9, 3])
    child_a, child_b = route_a.order_crossover(route_b)
    print(child_a.chromosome)
    print(child_b.chromosome)       