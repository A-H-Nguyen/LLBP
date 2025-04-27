import argparse
import json
import random
import copy
import re
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, 
                    help="Input config JSON file. NOTE: This file will be modified by this script!")
parser.add_argument("-w", "--warmup_instructions", type=int, required=True, 
                    help="Number of warmup instructions")
parser.add_argument("-n", "--simulation_instructions", type=int, required=True, 
                    help="Number of instructions of the region of interest (ROI)")
parser.add_argument("-p", "--population", type=int, default=20, required=False, 
                    help="Number of random configs generated at the beginning of tuning")
parser.add_argument("-g", "--generations", type=int, default=10, required=False, 
                    help="Number of iterations that the genetic algorithm will be run for")
parser.add_argument("-m", "--mutation_rate", type=float, default=0.2, required=False, 
                    help="Determines how often mutations occur per each generation")
parser.add_argument("-e", "--number_of_elites", type=int, default=2, required=False, 
                    help="Determines how many of the best individuals will be kept between generations")
parser.add_argument("--debug", action='store_true', help="Print debugging info")
args = parser.parse_args()

trace_dir="./traces/"
trace_ext = ".champsim.trace.gz"
# workloads = ["benchbase-tpcc",
#              "benchbase-twitter",
#              "benchbase-wikipedia",
#              "charlie.1006518",
#              "dacapo-kafka",
#              "dacapo-spring",
#              "dacapo-tomcat",
#              "mwnginxfpm-wiki",
#              "nodeapp-nodeapp",
#              "renaissance-finagle-chirper",
#              "renaissance-finagle-http"]
             # These traces failed to install -- will have to try again later:
             # "delta.507252", 
             # "merced.467915", 
             # "whiskey.426708"]

traces = []
for filename in os.listdir(trace_dir):
    filepath = os.path.join(trace_dir, filename)
    if filename.endswith(".gz") and os.path.isfile(filepath):
        traces.append(filepath)

initial_mpki = {trace:0.0 for trace in traces}

power_of_two_params = ["numPatterns", "numContexts", "ctxAssoc", "ptrnAssoc", "pbSize", "pbAssoc"]

param_def = {
    "numPatterns"     : (4,64),     # default: 16
    "numContexts"     : (1024,8192),# default: 4096
    "ctxAssoc"        : (1,16),     # default: 8
    "ptrnAssoc"       : (1,16),     # default: 4
    "TTWidth"         : (1,32),     # default: 13
    "CTWidth"         : (1,32),     # default: 14
    "pbSize"          : (16,256),   # default: 64
    "pbAssoc"         : (1,16),     # default: 4
    "CtrWidth"        : (1,32),     # default: 3
    "ReplCtrWidth"    : (1,32),     # default: 16
    "CtxReplCtrWidth" : (1,32),     # default: 2
    "accessDelay"     : (1,32)      # default: 5
}

def get_mpki():
    with open("output.log", 'r') as f:
        for line in f:
            if "ROI MPKI" in line:
                # Use regex to extract the floating point number
                match = re.search(r"ROI MPKI\s*:\s*([0-9.]+)", line)
                if match:
                    return float(match.group(1))
                else:
                    raise ValueError(f"MPKI value not found in line: {line}") 

def random_power_of_two(low, high):
    powers = []
    i = 0
    while (1 << i) <= high:
        val = 1 << i
        if val >= low:
            powers.append(val)
        i += 1
    return random.choice(powers)

def random_individual():
    indiv = {}
    for param in param_def.keys():
        low, high = param_def[param]
        if param in power_of_two_params:
            indiv[param] = random_power_of_two(low, high)
        else:
            indiv[param] = random.randint(low, high)
    return indiv

# def run_sim(individual):
#     """
#     We evaluate every individual by running all our traces on it.
#     Every individual gets the same number of warmup and simulation instructions.

#     This function returns a dictionary, so that every generation, we can see the
#     exact MPKI values for each individual (or the best individual).
#     """

def evaluate(individual):
    """
    After we run the simulator for every trace on an individual, we calculate
    the percent differences between its new MPKI values, and the baseline.
    Here, a positive difference means that MPKI *decreased* from the baseline,
    while a negative difference means that MPKI *increased*.

    Once we have all the MPKI differencs, we calculate the a weighted average,
    so that differences for larger initial MPKI values have a greater effect
    """
    weighted_sum = 0.0
    total_weight = 0.0
    # mpki_diff = {trace:0.0 for trace in traces}
    for trace in traces:
        with open("output.log", "w") as logfile:
            result = subprocess.run(["./build/predictor", "-c", args.config, 
                                     "-w", str(args.warmup_instructions), 
                                     "-n", str(args.simulation_instructions), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)
        new_mpki = get_mpki()
        if args.debug:
            print(f"New MPKI for {os.path.basename(trace)}:\t{new_mpki}")

        if new_mpki is None:
            # If the simulator fails, we penalize this individual by assigning it a massive
            # increase in MPKI
            # mpki_diff[trace] = -1e4
            weighted_sum += -1000.0 
            total_weight += 1.0 # to avoid divide by zero
            if args.debug:
                print(f"Simulator failed for trace {trace}, with config: {individual}")
        else:
            # mpki_diff[trace] = initial_mpki[trace] - new_mpki
            baseline = initial_mpki[trace]
            improvement = ((baseline - new_mpki) / baseline) * 100.0 # percent difference

            weight = baseline  # traces with higher baseline MPKI get bigger influence
            weighted_sum += improvement * weight
            total_weight += weight

    weighted_average_improvement = weighted_sum / total_weight
    return -weighted_average_improvement  # return a negative value because GA minimizes

def crossover(parent1, parent2):
    child = {}
    for param in param_def.keys():
        child[param] = random.choice([parent1[param], parent2[param]])
    return child

def mutate(individual, mutation_rate=0.1):
    mutant = copy.deepcopy(individual)
    for param in param_def.keys():
        if random.random() < mutation_rate:
            low, high = param_def[param]
            if param in power_of_two_params:
                mutant[param] = random_power_of_two(low, high)
            else:
                mutant[param] = random.randint(low, high)
    return mutant

def genetic_algorithm(pop_size, generations, mutation_rate, elite_size):
    # Initialize population
    population = [random_individual() for _ in range(pop_size)]
    
    for gen in range(generations):
        # Evaluate all individuals
        scored_population = [(evaluate(indiv), indiv) for indiv in population]
        for i in range(pop_size):
            while scored_population[i][0] is None:
                population[i] = random_individual()
                scored_population[i] = evaluate(population[i]), population[i]

        scored_population.sort(key=lambda x: x[0])  # Sort by MPKI (lower is better)

        print(f"Generation {gen}: Best MPKI = {scored_population[0][0]}")

        # Select elites
        elites = [indiv for _, indiv in scored_population[:elite_size]]

        # New population
        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(population)  # Can crossover with any
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    # Final evaluation
    final_scored_population = [(evaluate(indiv), indiv) for indiv in population]
    final_scored_population.sort(key=lambda x: x[0])

    best_mpki, best_individual = final_scored_population[0]
    print(f"Best individual after {generations} generations:\n{best_individual}\nwith MPKI {best_mpki}")
    return best_individual

if __name__ == "__main__":

    print("=" * 55, "\n")

    print("*** Genetic Algorithm parameters ***\n")
    print(f" - Population size: {args.population} individuals")
    print(f" - Number of generations: {args.generations}")
    print(f" - Mutation rate: {args.mutation_rate}")
    print(f" - Number of elites per generation: {args.number_of_elites}")

    print("=" * 55, "\n")

    print("*** Workloads being tested ***\n")
    for trace in traces:
        print(os.path.basename(trace))

    print("=" * 55, "\n")

    print("*** Initial MPKI values ***\n")
    for trace in traces:
        with open("output.log", "w") as logfile:
            result = subprocess.run(["./build/predictor", "-c", "configs/default_config.json", 
                                     "-w", str(args.warmup_instructions), 
                                     "-n", str(args.simulation_instructions), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)
        new_mpki = get_mpki()
        initial_mpki[trace] = new_mpki 
        print(f" - {os.path.basename(trace)}:\t{new_mpki}")

    print("=" * 55, "\n")

    genetic_algorithm(pop_size=args.population, generations=args.generations,
                      mutation_rate=args.mutation_rate, elite_size=args.number_of_elites)

