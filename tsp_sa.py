import math
import time
import random
import matplotlib.pyplot as plt

progress = []

def update_distance(route, matrix):
    distance = 0
    
    from_index = route[0]
    for i in range(len(route)):
        distance += matrix[from_index][route[i]]
        from_index = route[i]

    distance += matrix[from_index][route[0]]

    return distance

def calculate_distance(city_1, city_2):
    return int(math.sqrt(((city_1[0]-city_2[0])**2)+((city_1[1]-city_2[1])**2)))

def get_distances(cities_coordinates):
    distances = [ [] for _ in range(len(cities_coordinates)) ] 

    for i in range(len(cities_coordinates)):
        for j in range(len(cities_coordinates)):
            distances[i].append(calculate_distance(cities_coordinates[i], cities_coordinates[j]))

    return distances

def get_first_solution(cities_coordinates, distances):

    solution = []

    for i in range(len(cities_coordinates)):
        solution.append(i)

    random.shuffle(solution)

    current_fitness = update_distance(solution, distances)
  
    return solution, current_fitness
    
def simulated_annealling(cities_coordinates, distances, stop):
    alpha = 0.997
    stopping_temperature = 0
    temperature = 10000
    best_solution = None
    best_fitness = None

    # vygenerujeme prve nahodne riesenie
    current_solution, current_fitness = get_first_solution(cities_coordinates, distances)

    best_fitness = current_fitness
    best_solution = current_solution
    progress.append(current_fitness)
    
    iteration = 1

    while temperature >= stopping_temperature and iteration < stop:
            
            # vygenerujeme susednu trasu obratenim nahodneho useku
            candidate = current_solution.copy()
            l = random.randint(2, len(cities_coordinates) - 1)
            i = random.randint(0, len(cities_coordinates) - 1)
            candidate[i : (i + l)] = reversed(candidate[i : (i + l)])


            candidate_fitness = update_distance(candidate, distances)
            #ak je lepsi akceptujeme s p = 1
            if candidate_fitness < current_fitness:
                current_fitness, current_solution = candidate_fitness, candidate
                if candidate_fitness < best_fitness:
                    best_fitness, best_solution = candidate_fitness, candidate
            else:  #akceptujeme s pravdepodobnostou
                if random.random() < math.exp(-abs(candidate_fitness - current_fitness) / temperature):
                    current_fitness, current_solution = candidate_fitness, candidate
            # znizime teplotu
            temperature *= alpha
            iteration += 1

            progress.append(current_fitness)

    print("Total distance: ",best_fitness)

    return best_solution

if __name__ == "__main__":
    #coordinates = [(random.uniform(0, 200), random.uniform(0, 200)) for i in range(40)]    
    coordinates = [(60, 200), (180, 200), (100, 180), (140, 180), (20, 160), (80, 160), (200, 160), (140, 140), (40, 120), (120, 120), (180, 100), (60, 80), (100, 80), (180, 60), (20, 40), (100, 40), (200, 40), (20, 20), (60, 20),(160, 20)]

    distances = get_distances(coordinates)
    start_time = time.time()
    solution = simulated_annealling(coordinates, distances, 5000)
    print("Search time %s seconds" % (time.time() - start_time))
   
    print(solution[-1], end='')

    for i in range(0, len(solution)):
       print(' -> ', solution[i], end='')

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Iteration')
    plt.show()