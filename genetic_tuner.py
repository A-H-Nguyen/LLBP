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

def run_sim(individual):
    """
    We evaluate every individual by running all our traces on it.
    Every individual gets the same number of warmup and simulation instructions.

    This function returns a dictionary, so that every generation, we can see the
    exact MPKI values for each individual (or the best individual). We also calculate
    the percent difference from the initial values for every MPKI we gather here.
    """
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    for param in individual:
        config[param] = individual[param]
        
    with open(args.config, "w") as f:
        json.dump(config, f, indent=2)

    mpki = {trace:(0.0,0.0) for trace in traces}
    for trace in traces:
        with open("output.log", "w") as logfile:
            result = subprocess.run(["./build/predictor", "-c", args.config, 
                                     "-w", str(args.warmup_instructions), 
                                     "-n", str(args.simulation_instructions), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)
        new_mpki = get_mpki()

        if args.debug:
            print(f"New MPKI for {os.path.basename(trace)}:\t{new_mpki}\n")

        # If the simulator fails, we penalize this individual by assigning it a massive MPKI
        if new_mpki is None:
            new_mpki = 1e4
            if args.debug:
                print(f"Simulator failed for trace {trace}, with config: {individual}\n")

        mpki[trace] = new_mpki

    return mpki

def evaluate(sim_out):
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

    for trace,mpki in sim_out.items():
        baseline = initial_mpki[trace]
        improvement = ((baseline - mpki) / baseline) * 100.0 # percent difference

        if args.debug:
            print(f" - {os.path.basename(trace)}:\t{improvement }% diff")

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
    population = {f"gen0_indiv{i}":random_individual() for i in range(pop_size)}
    
    for gen in range(1,generations+1):
        # Generate sim results for all individuals
        mpki_out = {indiv_name:run_sim(indiv_conf) for indiv_name,indiv_conf in population.items()}

        # Evaluate all individuals
        scored_population = {indiv_name:evaluate(sim_out) for indiv_name,sim_out in mpki_out.items()}

        # Sort by difference in MPKI -- lower is better, with negative values meaning improvement
        # in MPKI overall.
        sorted_indivs = sorted(scored_population, key=lambda name: scored_population[name])

        best_indiv = sorted_indivs[0]

        print(f"\n*** Generation {gen} ***\n")
        print(f"Best Config - {best_indiv }:\n{population[best_indiv]}\n")
        for trace,mpki in mpki_out[best_indiv].items():
            print(f" - {os.path.basename(trace)}:\t{mpki} MPKI")
        print(f"\nOverall improvement: {scored_population[best_indiv]}\n")

        elite_names = sorted_indivs[:elite_size]
        elite_population = {name: population[name] for name in elite_names}

        next_population = {}
        for name, config in elite_population.items():
            next_population[name] = config

        while len(next_population) < pop_size:
            # Assign a new name for the child
            child_id = f"gen{gen}_indiv{len(next_population)}"

            # Randomly select parents from the elites
            parent1_name = random.choice(list(elite_population.keys()))
            parent2_name = random.choice(list(elite_population.keys()))
            
            parent1 = elite_population[parent1_name]
            parent2 = elite_population[parent2_name]

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)

            next_population[child_id] = child

        population = next_population

    # Final evaluation
    final_mpki_out = {indiv_name:run_sim(indiv_conf) for indiv_name,indiv_conf in population.items()}
    final_scored_population = {indiv_name:evaluate(sim_out) for indiv_name,sim_out in mpki_out.items()}
    final_sorted_indivs = sorted(final_scored_population , key=lambda name: final_scored_population[name])
    final_best_indiv = final_sorted_indivs[0]

    print(f"\n*** Best individual after {generations} generations ***\n")
    print(f"Final conf: {final_best_indiv}\n{population[final_best_indiv]}\n")
    for trace,mpki in final_mpki_out[final_best_indiv].items():
        print(f" - {os.path.basename(trace)}:\t{mpki} MPKI")
    print(f"\nOverall improvement: {final_scored_population[final_best_indiv]}\n")


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
        print(" -", os.path.basename(trace))

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

    # test_indiv = random_individual()
    # print("Testing config:", test_indiv)
    # sim_out = run_sim(test_indiv )
    # print()
    # print(evaluate(sim_out))

    genetic_algorithm(pop_size=args.population, generations=args.generations,
                      mutation_rate=args.mutation_rate, elite_size=args.number_of_elites)

