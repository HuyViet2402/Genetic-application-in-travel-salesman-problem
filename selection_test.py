import random
import time
import numpy as np
from itertools import permutations
POPULATION_SIZE = 1000
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
    #population = sorted(population, key=lambda x: x.fitness) # must sorted in descending order in fitness for the probability of being chosen increase
    fitness_scores = [1/individual.fitness for individual in population]
    total_fitness = sum(fitness_scores)
    probability_individual = [fitness/total_fitness for fitness in fitness_scores]
    return np.random.choice(population, p=probability_individual)
    # roll = random.uniform(1/population[0].fitness, total_fitness)
    # cumulative_probability = 0
    # for individual in population:
    #     cumulative_probability += 1/individual.fitness
    #     if cumulative_probability >= roll:
    #         print(f"Chosen individual: {individual.chromosome}")
    #         print(f"Fitness of chosen individual: {individual.fitness}")
    #         print(f"Roll: {(roll/total_fitness*100)}%")
    #         print(f"Probability of chosen individual(Cumulative probability): {(cumulative_probability/total_fitness)*100}%")
    #         break
    #         return individual
    #     print(f"Individual: {individual.chromosome}")
    #     print(f"Fitness: {individual.fitness}")
    #     print(f"Probability of individual: {(cumulative_probability/total_fitness)*100}%")
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
        #     print(f"Chosen individual: {population[i].chromosome}")
        #     print(f"Chosen rank: {i+1}")
        #     print(f"Fitness of chosen individual: {population[i].fitness}")
        #     print(f"Roll: {roll}")
        #     print(f"Probability of chosen rank(Cumulative probability): {cumulative_probability}")
        #     break
            return population[i]
        # print(f"Rank: {i+1}")
        # print(f"Probability of rank {i+1}: {probability[i]}")
        # print(f"Fitness of individual: {population[i].fitness}")
    
def tournament_selection(population, k=10):
    """ Select k individuals from parents
        Select the best individual from k
        n is population size
        Time complexity: O(1)
        Space complexity: O(1)
    """
    start = random.randint(0, POPULATION_SIZE - k)
    end = start + k
    return min(population[start:end], key=lambda x: x.fitness)

def find_most_repeated_element(lst):
    """Finds the element with the highest count in a list.

    Args:
        lst: The input list.

    Returns:
        A tuple containing the most repeated element and its count.
    """

    count_dict = {}
    for element in lst:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1

    max_count = max(count_dict.values())

    return max_count
population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]
def tournament_test():
    new_population = []
    for _ in range(POPULATION_SIZE):
        new_population.append(tournament_selection(population))
    
    most_repeted_time = find_most_repeated_element(new_population)
    print(f"Percentage of time the tournament selection return same individual: {(most_repeted_time/POPULATION_SIZE)*100}%")

def rank_test():
    new_population = []
    for _ in range(POPULATION_SIZE):
        new_population.append(rank_selection(population))
    
    most_repeted_time = find_most_repeated_element(new_population)
    print(f"Percentage of time the rank selection return same individual: {(most_repeted_time/POPULATION_SIZE)*100}%")

def roulette_wheel_test():
    new_population = []
    for _ in range(POPULATION_SIZE):
        new_population.append(roulette_wheel(population))
    
    most_repeted_time = find_most_repeated_element(new_population)
    print(f"Percentage of time the roulette wheel selection return same individual: {(most_repeted_time/POPULATION_SIZE)*100}%")

if __name__ == '__main__':
    # for i in population:
    #     print (i.chromosome)
    rank_test()
    roulette_wheel_test()
    #rank_selection(population)
    #roulette_wheel(population)