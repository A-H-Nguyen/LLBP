import argparse
import json
import random
import copy
import re
import subprocess
import os
import itertools
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, 
                    help="Input config JSON file. NOTE: This file will be modified by this script!")
parser.add_argument("-w", "--warmup_instructions", type=int, required=True, 
                    help="Number of warmup instructions")
parser.add_argument("-n", "--simulation_instructions", type=int, required=True, 
                    help="Number of instructions of the region of interest (ROI)")
parser.add_argument("--limit", type=int, required=True, 
                    help="Max number of algorithm iterations")
parser.add_argument("--debug", action='store_true', help="Print debugging info")
args = parser.parse_args()

with open("output/baseline_mpki.json", "r") as f:
    initial_mpki = json.load(f)

trace_dir="./traces/"
trace_ext = ".champsim.trace.gz"
training_workloads = [
        "benchbase-tpcc", 
        "benchbase-twitter", 
        "benchbase-wikipedia",
        "charlie.1006518", 
        "dacapo-kafka", 
        "dacapo-tomcat",
        "mwnginxfpm-wiki", 
        "renaissance-finagle-http",
        "merced.467915", 
        "whiskey.426708"
        ]
testing_workloads = [
        "nodeapp-nodeapp", 
        "dacapo-spring", 
        "delta.507252", 
        "renaissance-finagle-chirper"
        ]

train_traces = [trace_dir + workload + trace_ext for workload in training_workloads]

power_of_two_params = ["numPatterns", "numContexts", "ctxAssoc", "ptrnAssoc", "pbSize", "pbAssoc"]

param_def = {
    "numPatterns"     : [2**i for i in range(2,9)],  # default: 16
    "numContexts"     : [2**i for i in range(10,14)],# default: 4096
    "ctxAssoc"        : [2**i for i in range(1,5)],  # default: 8
    "ptrnAssoc"       : [2**i for i in range(1,5)],  # default: 4
    "TTWidth"         : [i for i in range(10,15)],   # default: 13
    "CTWidth"         : [i for i in range(10,15)],   # default: 14
    "pbSize"          : [2**i for i in range(4,8)],  # default: 64
    "pbAssoc"         : [2**i for i in range(0,5)],  # default: 4
    "CtrWidth"        : [i for i in range(1,6)],     # default: 3
    "ReplCtrWidth"    : [2**i for i in range(0,6)],  # default: 16
    "CtxReplCtrWidth" : [2**i for i in range(0,6)]   # default: 2
}

default_conf = {
    "numPatterns"     : 16,
    "numContexts"     : 4096,
    "ctxAssoc"        : 8,
    "ptrnAssoc"       : 4,
    "TTWidth"         : 13,
    "CTWidth"         : 14,
    "pbSize"          : 64,
    "pbAssoc"         : 4,
    "CtrWidth"        : 3,
    "ReplCtrWidth"    : 16,
    "CtxReplCtrWidth" : 2
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


def run_sim(individual):
    """
    We evaluate every individual by running all our traces on it.
    Every individual gets the same number of warmup and simulation instructions.

    This function returns a dictionary, so that every generation, we can see the
    exact MPKI values for each individual (or the best individual). 
    """
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    for param in individual:
        config[param] = individual[param]
        
    with open(args.config, "w") as f:
        json.dump(config, f, indent=2)

    mpki = {trace:(0.0,0.0) for trace in train_traces}
    for trace in train_traces:
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
    return weighted_average_improvement 



if __name__ == "__main__":
    random.seed(0)

    # Generate search space
    keys, values = zip(*param_def.items())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    num_configs = len(grid)

    # Reset output files
    with open("output/grid_search_dump.log", "w") as f:
        print("=" * 55, "\n")
        print("*** Simulation parameters ***\n")
        print(f" - Num warmup instructions: {args.warmup_instructions}")
        print(f" - Num simulation instructions: {args.simulation_instructions}")
        print( " - Workloads used for training:")
        f.write("=" * 55)
        f.write("\n")
        f.write("*** Simulation parameters ***\n")
        f.write(f" - Num warmup instructions: {args.warmup_instructions}\n")
        f.write(f" - Num simulation instructions: {args.simulation_instructions}\n")
        f.write( " - Workloads used for training:\n")
        for trace in train_traces:
            print("   -", os.path.basename(trace))
            f.write(f"   - {os.path.basename(trace)}\n")

        print("=" * 55, "\n")
        f.write("=" * 55)
        f.write("\n")

        print(f"*** {num_configs} configurations in testing set")
        print(f"*** Evaluate {args.limit} of them\n")
        f.write(f"*** {num_configs} configurations in testing set\n")
        f.write(f"*** Evaluate {args.limit} of them\n")

    with open("output/grid_search_iterations.csv", "w") as f:
        f.write("ITERATION NUMBER,MPKI IMPROVEMENT,BEST IMPROVEMENT,RUNTIME(sec)\n")
        print("ITERATION NUMBER,MPKI IMPROVEMENT,BEST IMPROVEMENT,RUNTIME(sec)")

    best_conf = default_conf.copy()
    best_conf_train_results = {}
    test_conf = {}
    best_eval = 0.0
    iter_num = 0

    iter_file = open("output/grid_search_iterations.csv", "a")
    dump_file = open("output/grid_search_dump.log", "a")

    while iter_num < args.limit:
        start_time = time.perf_counter()
        test_conf = grid.pop(random.randint(0,num_configs-1)).copy()
        sim_out = run_sim(test_conf)
        new_eval = evaluate(sim_out)

        if new_eval > best_eval:
            best_eval = new_eval
            best_conf = test_conf.copy()
            best_conf_train_results = sim_out.copy()

            print("ITERATION:", iter_num + 1)
            print("New best conf:")
            print(best_conf)
            print("New best train results:")
            print(best_conf_train_results, '\n')

            dump_file.write(f"ITERATION: {iter_num + 1}\n")
            dump_file.write("New best conf:\n")
            dump_file.write(best_conf)
            dump_file.write("\n")
            dump_file.write("New best train results:\n")
            dump_file.write(best_conf_train_results)
            dump_file.write("\n\n")

        end_time = time.perf_counter()
        iter_num += 1
        num_configs -=1

        print(f"{iter_num},{new_eval},{best_eval},{end_time - start_time}")
        iter_file.write(f"{iter_num},{new_eval},{best_eval},{end_time - start_time}\n")

