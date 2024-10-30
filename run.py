
import value
import TSP
import time
import test_case
import random
if __name__=="__main__":
    correct_path = 0
    brute_force_time_performance = 0
    genetic_time_performance = 0
    for i in range(value.TESTS):
        #random.seed(i)
        distance_matrix = test_case.random_testcase[i]
        gene_start = time.perf_counter()
        gene = TSP.genetic_algorithm(distance_matrix)
        gene_end = time.perf_counter()
        print(f"Genetic algorithm runtime: {gene_end - gene_start:0.5f} seconds")
        genetic_time_performance += (gene_end - gene_start)

        brute_force_start = time.perf_counter()
        brute_force = TSP.brute_force_algorithm(distance_matrix)
        brute_force_end = time.perf_counter()
        print(f"Brute force algorithm runtime: {brute_force_end - brute_force_start:0.5f} seconds")
        brute_force_time_performance += (brute_force_end - brute_force_start)

        if(gene == brute_force):
            correct_path+=1

    print(f"Percentage of correct path found by genetic algorithm: {(correct_path/value.TESTS) * 100}%")
    print(f"Total brute force algorithm runtime: {brute_force_time_performance:0.5f} seconds")
    print(f"Total genetic algorithm algorithm runtime: {genetic_time_performance:0.5f} seconds")