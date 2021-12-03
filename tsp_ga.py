import random
import copy
import math
import matplotlib.pyplot as plt
from numpy.random import choice
import time

progress = []

class State:
    
    def __init__(self, route, distance = 0):
        self.route = route
        self.distance = distance
    
    def __lt__(self, other):
         return self.distance < other.distance

    def deepcopy(self):
        return State(copy.deepcopy(self.route), copy.deepcopy(self.distance))

    # vypocita dlzku cesty
    def update_distance(self, matrix):
        
        self.distance = 0
        
        from_index = self.route[0]
        for i in range(len(self.route)):
            self.distance += matrix[from_index][self.route[i]]
            from_index = self.route[i]
        # vzdialenost do 1. mesta
        self.distance += matrix[from_index][self.route[0]]

def calculate_distance(city_1, city_2):
    return int(math.sqrt(((city_1[0]-city_2[0])**2)+((city_1[1]-city_2[1])**2)))

def get_distances(cities_coordinates):
    distances = [ [] for _ in range(len(cities_coordinates)) ] 

    for i in range(len(cities_coordinates)):
        for j in range(len(cities_coordinates)):
            distances[i].append(calculate_distance(cities_coordinates[i], cities_coordinates[j]))

    return distances

def create_first_generation(matrix, coordinates, size):
    
    gene_pool = []
    
    for i in range(len(coordinates)):
        gene_pool.append(i)
    
    population = []
    
    for i in range(size):

        #nahodne usporiada geny v jedincovi    
        random.shuffle(gene_pool)
       
        state = State(gene_pool)
        state.update_distance(matrix)
        
        population.append(state)
    
    return population

def breed(parents):
    part_1 = []
    part_2 = []
    parent_1 = parents[0].deepcopy()
    parent_2 = parents[1].deepcopy()
    
    a = int(random.random() * len(parent_1.route))
    b = int(random.random() * len(parent_1.route))
    start_gene = min(a, b)
    end_gene = max(a, b)
    # Geny z rodica 1
    for i in range(start_gene, end_gene):
        part_1.append(parent_1.route[i])
    
    # geny z rodica 2
    part_2 = [int(x) for x in parent_2.route if x not in part_1]

    child = [None] * (len(part_1) + len(part_2))

    for i in range(0, len(part_1)):
        child[start_gene] = part_1[i]
        start_gene = start_gene + 1
    
    k = 0
    for j in range(0, len(part_1)+ len(part_2)):
        if None is child[j]:
            child[j] = part_2[k]
            k += 1

    return child

def crossover(matrix, parents):
    
    state = State(breed(parents))
    state.update_distance(matrix)
    
    return state

def mutate(matrix, state, mutation_rate):
    
    mutated_state = state.deepcopy()

    for i in range(len(mutated_state.route)):
        
        if(random.random() < mutation_rate):
    
            j = int(random.random() * len(state.route))
            city_1 = mutated_state.route[i]
            city_2 = mutated_state.route[j]
            mutated_state.route[i] = city_2
            mutated_state.route[j] = city_1
    
    mutated_state.update_distance(matrix)
    
    return mutated_state

def tournament_selection(generation, parents_size = 50):
    
    population = generation.copy()
    selected = []
    parents = []

    i = 0
    while i < parents_size: 
        
        # vyber jedincov do turnaja
        tournament_participants = random.choices(population, k=5)
        tournament_participants.sort()
        
        # pridanie vyhercu medzi rodicov
        if(tournament_participants[0] not in selected):
            selected.append(tournament_participants[0])
            population.remove(tournament_participants[0])
            i += 1

    for j in range(1, len(selected)):
            parents.append((selected[j-1], selected[j]))
    
    return parents

def roullette_wheel_selection(generation, parents_size = 50):
    
    population = generation.copy()
    selected = []
    fitnessResults = []
    parents = []
    
    for i in range(0, len(population)):
        fitnessResults.append(1 / population[i].distance)

    total_fitness = sum(fitnessResults)

    probability_list = [f/total_fitness for f in fitnessResults]
    
    i = 0
    
    while i < parents_size: 
        winner = choice(population, p=probability_list)
       
        if(winner not in selected):
            selected.append(winner)
            i += 1
    
    for j in range(1, len(selected)):
            parents.append((selected[j-1], selected[j]))
    
    return parents
    
def elitism(population, elite_size=10):
    
    population = population[:elite_size]
    
    return population

def genetic_algorithm(matrix, population, mutation_rate, generations):
    
    # vytvori nove generacie
    for i in range(generations):
        
        # zoradi populaciu
        population.sort()
        progress.append(population[0].distance)
        
        parents = []
        
        # vyber rodicov
        parents = tournament_selection(population)
        
        children = []
        # krizenie rodicov
        for partners in parents:
            children.append(crossover(matrix, partners))
        # mutacie potomkov
        for j in range(len(children)):
            children[j] = mutate(matrix, children[j], mutation_rate)

        #vytvorenie novej populacie pridanim elitarov a novych potomkov
        population = elitism(population)
        population.extend(children)
    
    population.sort()
    
    return population[0]
    
def main():
    #coordinates = [(random.uniform(0, 200), random.uniform(0, 200)) for i in range(40)]
    coordinates = [(60, 200), (180, 200), (100, 180), (140, 180), (20, 160), (80, 160), (200, 160), (140, 140), (40, 120), (120, 120), (180, 100), (60, 80), (100, 80), (180, 60), (20, 40), (100, 40), (200, 40), (20, 20), (60, 20), (160, 20)]
    
    distances = get_distances(coordinates)
    
    start_time = time.time()
    population = create_first_generation(distances, coordinates, 100)
    state = genetic_algorithm(distances, population, 0.01, 100)
    print("Search time %s seconds" % (time.time() - start_time))
    print(state.route[-1], end='')
    for i in range(0, len(state.route)):
       print(' -> ', state.route[i], end='')
    
    print('\nTotal distance: ', state.distance)

    plt.plot(progress)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()

if __name__ == "__main__": main()