import value
import random
class Individual:
    ''' Class representing individual in the population (a tour) '''

    def __init__(self, chromosome, distance_matrix):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness(distance_matrix)

    @classmethod
    def create_gnome(self):
        ''' Create a random chromosome (tour) by shuffling cities '''
        gnome = list(range(value.CITIES))
        random.shuffle(gnome)
        return gnome
    def calculate_fitness(self, distance_matrix):
        ''' Calculate fitness as the total distance of the tour '''
        total_distance = 0
        for i in range(value.CITIES - 1):
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