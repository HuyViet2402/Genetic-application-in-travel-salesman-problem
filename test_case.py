import random
import value
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

random_testcase = []
for i in range(value.TESTS):
    random.seed(i)
    distance_matrix = create_graph(value.CITIES, 1, 10000)
    random_testcase.append(distance_matrix)