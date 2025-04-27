import argparse
import json
import random
import copy
import re
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-w", "--warmup_instructions", type=int, required=True)
parser.add_argument("-n", "--simulation_instructions", type=int, required=True)
args = parser.parse_args()

trace_dir="./traces/"
workloads=["mwnginxfpm-wiki", "dacapo-kafka", "dacapo-tomcat", "dacapo-spring", 
        "renaissance-finagle-chirper", "renaissance-finagle-http"]
        # "benchbase-tpcc", 
        # "benchbase-twitter", "benchbase-wikipedia", "nodeapp-nodeapp", "charlie.1006518", 
        # "delta.507252", "merced.467915", "whiskey.426708"]
trace_ext = ".champsim.trace.gz"

power_of_two_params = ["numPatterns", "numContexts", "ctxAssoc", "ptrnAssoc", "pbSize", "pbAssoc"]

param_def = {
    "numPatterns"     : (4,64),     # default: 16
    "numContexts"     : (1024,8192),# default: 4096
    "ctxAssoc"        : (1,16),     # default: 8
    "ptrnAssoc"       : (1,16),     # default: 4
    "TTWidth"         : (1,32),     # default: 13
    "CTWidth"         : (1,32),     # default: 14
    "pbSize"          : (16,256),    # default: 64
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

def evaluate(individual):
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    for param in individual:
        config[param] = individual[param]
        
    with open(args.config, "w") as f:
        json.dump(config, f, indent=2)

    mpki_values = []
    for workload in workloads:
        trace = trace_dir + workload + trace_ext
        with open("output.log", "w") as logfile:
            result = subprocess.run(["./build/predictor", "-c", args.config, 
                                     "-w", str(args.warmup_instructions), 
                                     "-n", str(args.simulation_instructions), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)

        new_mpki = get_mpki()
        if new_mpki is None:
            # print(f"Simulator failed for trace {trace}\nwith config: {individual}")
            mpki_values.append(1e6)
        else:
            mpki_values.append(new_mpki)

    return sum(mpki_values) / len(mpki_values)


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

def genetic_algorithm(pop_size=20, generations=50, mutation_rate=0.1, elite_size=2):
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
    population = 20
    generations = 10

    for workload in workloads:
        trace = trace_dir + workload + trace_ext
        with open("output.log", "w") as logfile:
            result = subprocess.run(["./build/predictor", "-c", "configs/default_config.json", 
                                     "-w", str(args.warmup_instructions), 
                                     "-n", str(args.simulation_instructions), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)
        print(f"Default MPKI for {workload}:\t{get_mpki()}")
    print("=" * 55)
    print()

    genetic_algorithm(pop_size=population, generations=generations)
    # output = evaluate(random_individual())
    # while output is None:
    #     output = evaluate(random_individual())
