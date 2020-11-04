''' File Information
For COMP3211 Fundamentals of Artificial Intelligence
Assignment1 Problem3 at HKUST in 2020 Fall.

Please `pip install --upgrade numpy` if numpy is not installed
Please put this file and gp-training-set.csv in the same directory.
'''
import sys
import numpy as np

POPULATION_SIZE = 1000
MAX_ITERATIONS = 1000
MUTATION_PROB = 5 # 0 ~ 1000

# Debugging Tool
def eprint(message):
    sys.stderr.write(message)
    sys.stderr.write('\n')
    sys.stderr.flush()

# Load the training set
data = np.genfromtxt('gp-training-set.csv', delimiter=',')
n = data.shape[1] - 1

training_data = np.ones((data.shape[0], n+1), dtype=np.float)
training_data[:, 1:] = data[:, 0:n]
# print(training_data)
training_data_target = data[:, [n]]
# print(training_data_target)
# print(type(training_data[0]))
#----------

# Generation 0
population = np.random.randn(POPULATION_SIZE, n+1)
# A hundred (w0, w1, ..., wn) where w0 = -theta, threshold then be 0.
# print(population[0])

def normalize(input):
    maxNum = abs(input.max())
    minNum = abs(input.min())

    if maxNum == 0:
        maxNum = 0.01
    if minNum == 0:
        minNum = 0.01

    boundary = 0.01
    if maxNum > minNum:
        boundary = maxNum
    else:
        boundary = minNum
    
    input /= boundary
    return None

for i in range(POPULATION_SIZE):
    normalize(population[i])
# print(population[0])
#----------

# Repeat until miss only 5% or reaching 1000 iterations
def fitness(program):
    accurate_case = 0
    count = 0
    for row in training_data:
        target = training_data_target[count]
        output = 0.0
        if np.dot(program, row) >= 0:
            output = 1.0
        if target == output:
            accurate_case += 1

        count += 1
    
    accuracy = float(accurate_case)/count
    return accuracy

# Debugging Tool
def debug(population):
    eprint("Debug...")
    for program in population:
        eprint("Accuracy = {}".format(fitness(program)))

def selection(population):
    def predicate(input_population):
        return np.array([fitness(program) for program in input_population])
    
    order = np.argsort(predicate(population))[::-1]
    return population[order]

def exchange(programA, programB, position):
    programA_new = np.concatenate((programA[:position], programB[position:]))
    programB_new = np.concatenate((programB[:position], programA[position:]))
    return programA_new, programB_new

def crossover(population):
    # selecting first 20% and in pairs
    chunk_size = int(POPULATION_SIZE/5)
    for j in range(int(POPULATION_SIZE/5/2)):
        # five breaking points, while keep the good ones at the same time (the first 20)
        positions = np.random.randint(low=0, high=len(population[2*j]), size=(5-1))
        counter = 5
        for position in positions:
            counter -= 1
            population[2*j+chunk_size*counter], population[2*j+1+chunk_size*counter] = \
                exchange(population[j], population[j+1], position)

def mutation(population):
    for idx, program in enumerate(population):
        prob = np.random.randint(low=0,high=1000)
        if prob < MUTATION_PROB:
            position = np.random.randint(low=0, high=len(program))
            program[position] = (0 - program[position])
            # eprint("A Mutation Occurs at program:{}, position:{}".format(idx, position))

for i in range(MAX_ITERATIONS):
    if i%20 == 0:
        eprint("Round {}".format(i))
        eprint("Accuracy = {}".format(fitness(population[0])))
    
    # Selection
    population = selection(population)
    # select the first 20% implicitly
    # eprint("Selection")
    # debug(population)

    # Check if the condition "error rate < 5%" meets
    if (1 - fitness(population[0])) < 0.05:
        break

    # Crossover
    crossover(population)
    # eprint("Crossover")
    # debug(population)

    # Mutation
    mutation(population)
    # eprint("Mutation")
    # debug(population)

    # End repeat
#----------

# Print out the best one from (w0, ..., wn) to (w1, ..., wn, theta)
print("(w1, ..., wn, theta) = (", end="")
for i in range(n):
    print("{}, ".format(population[0][i+1]), end="")
print("{})".format(-population[0][0]))

print("Accuracy = {}".format(fitness(population[0])))