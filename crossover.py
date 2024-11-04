import random
from individual import Individual
import numpy as np
################################### Crossover ################################################
def mate(partner1, partner2):
    ''' Perform crossover and produce offspring '''
    child_chromosome = []
    # Select a subset of cities from parent 1
    start = random.randint(0, len(partner1.chromosome) - 1)
    end = random.randint(start, len(partner1.chromosome) - 1)
    child_chromosome[start:end] = partner1.chromosome[start:end]
    # Fill the rest from parent 2, maintaining order
    for city in partner2.chromosome:
        if city not in child_chromosome:
            child_chromosome.append(city)
    return Individual(child_chromosome)

def one_point_crossover(partner1, partner2, distance_matrix):
    point =  random.randint(0, len(partner1.chromosome) - 1)
    child1 = partner1.chromosome[:point] + partner2.chromosome[point:]
    child2 = partner2.chromosome[:point] + partner1.chromosome[point:]
    return Individual(child1, distance_matrix), Individual(child2, distance_matrix)

# def one_point_crossover(partner1, partner2, distance_matrix):
#     # Generate a random crossover point
#     point = random.randint(0, len(partner1.chromosome) - 1)
#     # Create child templates from the crossover point
#     child1 = partner1.chromosome[:point]
#     child2 = partner2.chromosome[:point]
#     for gene in partner2.chromosome:
#         if gene not in child1:
#             child1.append(gene)
#     for gene in partner1.chromosome:
#         if gene not in child2:
#             child2.append(gene)
#     return Individual(child1, distance_matrix), Individual(child2, distance_matrix)

def uniform_crossover(partner1, partner2, distance_matrix):
    child1 = list(partner1.chromosome)
    child2 = list(partner2.chromosome)
    for i in range(len(partner1.chromosome)):
        if bool(random.getrandbits(1)):
            child1[i], child2[i] = child2[i], child1[i]
    return Individual(child1, distance_matrix), Individual(child2, distance_matrix)

# def uniform_crossover(partner1, partner2, distance_matrix):
#     # Initialize two empty chromosomes for the offspring
#     child1 = [None] * len(partner1.chromosome)
#     child2 = [None] * len(partner1.chromosome)
#     # Track genes already added to avoid duplicates
#     used_in_child1 = set()
#     used_in_child2 = set()
#     # Perform uniform crossover by iterating through each gene
#     for i in range(len(partner1.chromosome)):
#         if bool(random.getrandbits(1)):
#             # Swap genes between parents for child1 and child2
#             if partner2.chromosome[i] not in used_in_child1:
#                 child1[i] = partner2.chromosome[i]
#                 used_in_child1.add(partner2.chromosome[i])
#             if partner1.chromosome[i] not in used_in_child2:
#                 child2[i] = partner1.chromosome[i]
#                 used_in_child2.add(partner1.chromosome[i])
#         else:
#             # Keep the genes from the respective parents
#             if partner1.chromosome[i] not in used_in_child1:
#                 child1[i] = partner1.chromosome[i]
#                 used_in_child1.add(partner1.chromosome[i])
#             if partner2.chromosome[i] not in used_in_child2:
#                 child2[i] = partner2.chromosome[i]
#                 used_in_child2.add(partner2.chromosome[i])
#     # Fill in any missing genes to avoid None values
#     for i in range(len(partner1.chromosome)):
#         if child1[i] is None:
#             for gene in partner1.chromosome:
#                 if gene not in used_in_child1:
#                     child1[i] = gene
#                     used_in_child1.add(gene)
#                     break
#         if child2[i] is None:
#             for gene in partner2.chromosome:
#                 if gene not in used_in_child2:
#                     child2[i] = gene
#                     used_in_child2.add(gene)
#                     break
#     return Individual(child1, distance_matrix), Individual(child2, distance_matrix)

def cycle_crossover(partner1, partner2, distance_matrix):
    ''' Return 2 children
        1. start with the first gene of parent 1
        2. look at the gene at the same position of parent 2
        3. go to the position with the same gene in parent 1
        4. add this gene index to the cycle
        5. repeat the steps 2-5 until we arrive at the starting gene of this cycle
        m is length of the chromosome (number of cities)
        Time complexity: O(2m)
        Space complexity: O(2m)
    '''
    child1 = [None] * len(partner1.chromosome)
    child2 = [None] * len(partner1.chromosome)
    visited = set()
    start_index = 0
    current_index = start_index
    while current_index not in visited:
        visited.add(current_index)
        child1[current_index] = partner1.chromosome[current_index]
        child2[current_index] = partner2.chromosome[current_index]
        next_index = partner2.chromosome.index(partner1.chromosome[current_index])
        current_index = next_index
    for i in range(len(partner1.chromosome)):
        if i not in visited:
            child1[i] = partner2.chromosome[i]
            child2[i] = partner1.chromosome[i]
    return Individual(child1, distance_matrix), Individual(child2, distance_matrix)

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
def PMX_crossover(partner1, partner2, distance_matrix):
    start = random.randint(0, len(partner1.chromosome) - 1)
    end = random.randint(start, len(partner1.chromosome) - 1)
    # print("start:", start)
    # print("end:", end)
    def PMX_one_offspring(p1, p2):
        offspring = np.zeros(len(p1), dtype=p1.dtype)
        # Copy the mapping section (middle) from parent1
        offspring[start:end + 1] = p1[start:end + 1]
        # print(offspring)
        # copy the rest from parent2 (provided it's not already there
        for i in np.concatenate([np.arange(0, start), np.arange(end + 1, len(p1))]):
            candidate = p2[i]
            while candidate in p1[start:end + 1]:  # allows for several successive mappings
               # print(f"Candidate {candidate} not valid in position {i}")  # DEBUGONLY
                candidate = p2[np.where(p1 == candidate)[0][0]]
                # print(candidate)
            offspring[i] = candidate
        return offspring
    child1 = PMX_one_offspring(np.array(partner1.chromosome), np.array(partner2.chromosome))
    child2 = PMX_one_offspring(np.array(partner2.chromosome), np.array(partner1.chromosome))
    return Individual(child1.tolist(), distance_matrix), Individual(child2.tolist(), distance_matrix)

def order_crossover(partner1, partner2, distance_matrix):
    ''' Return 2 children
        1. Select a random swath of consecutive gene from parent 1
        2. Drop the swath down to Child 1 and mark out these gene in Parent 2.
        3. Starting on the right side of the swath, grab gene from parent 2 in order not in child 1 and insert them in Child 1 at the right edge of the swath. 
        4. flip Parent 1 and Parent 2 and go back to Step 1 to produce child 2
        m is length of the chromosome (number of cities)
        Time complexity: O(2m)
        Space complexity: O(2m)
    '''
    start = random.randint(0, len(partner1.chromosome) - 2)
    end = random.randint(start+1, len(partner1.chromosome) - 1) 
    child1 = [None] * len(partner1.chromosome)
    child2 = [None] * len(partner1.chromosome)
    
    child1[start:end+1] = partner1.chromosome[start:end+1]
    child2[start:end+1] = partner2.chromosome[start:end+1]
    startIndex = (end + 1) % len(partner1.chromosome)
    j = (end + 1) % len(partner1.chromosome)
    k = (end + 1) % len(partner1.chromosome)
    for i in range(startIndex, len(partner1.chromosome)):
        if partner2.chromosome[i] not in child1:
            child1[j] = partner2.chromosome[i] 
            j = (j + 1) % len(partner1.chromosome)
        if partner1.chromosome[i] not in child2:
            child2[k] = partner1.chromosome[i] 
            k = (k + 1) % len(partner1.chromosome)
    for i in range(0, startIndex):
        if partner2.chromosome[i] not in child1:
            child1[j] = partner2.chromosome[i] 
            j = (j + 1) % len(partner1.chromosome)
        if partner1.chromosome[i] not in child2:
            child2[k] = partner1.chromosome[i] 
            k = (k + 1) % len(partner1.chromosome)
    return Individual(child1, distance_matrix), Individual(child2, distance_matrix)
################################### Crossover ################################################