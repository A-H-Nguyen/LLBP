import argparse
import json
import random
import copy
import re
import subprocess
import os
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
# parser.add_argument("--debug", action='store_true', help="Print debugging info")
args = parser.parse_args()

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

baseline_mpki = {
        "nodeapp-nodeapp": 4.3452,
        "dacapo-spring": 3.6435,
        "delta.507252": 1.1535,
        "renaissance-finagle-chirper": 0.4979
        }
traces = [trace_dir + workload + trace_ext for workload in training_workloads]

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

# def evaluate(sim_out):
#     """
#     After we run the simulator for every trace on an individual, we calculate
#     the percent differences between its new MPKI values, and the baseline.
#     Here, a positive difference means that MPKI *decreased* from the baseline,
#     while a negative difference means that MPKI *increased*.

#     Once we have all the MPKI differencs, we calculate the a weighted average,
#     so that differences for larger initial MPKI values have a greater effect
#     """
#     weighted_sum = 0.0
#     total_weight = 0.0

#     for trace,mpki in sim_out.items():
#         baseline = initial_mpki[trace]
#         improvement = ((baseline - mpki) / baseline) * 100.0 # percent difference

#         if args.debug:
#             print(f" - {os.path.basename(trace)}:\t{improvement }% diff")

#         weight = baseline  # traces with higher baseline MPKI get bigger influence
#         weighted_sum += improvement * weight
#         total_weight += weight

#     weighted_average_improvement = weighted_sum / total_weight
#     return -weighted_average_improvement  # return a negative value because GA minimizes



if __name__ == "__main__":

    print("=" * 55, "\n")

    print("*** Simulation parameters ***\n")
    print(f" - Num warmup instructions: {args.warmup_instructions}")
    print(f" - Num simulation instructions: {args.simulation_instructions}")
    print( " - Workloads used for training:")
    for trace in traces:
        print("   -", os.path.basename(trace))

    print("=" * 55, "\n")

    iter = 0
    while iter <

    print(f"*** Evaluating final config ***\n") 

    EVAL_WARMUP = 100000000
    EVAL_SIM    = 200000000
    test_traces = [trace_dir + workload + trace_ext for workload in testing_workloads]

    print(f" - Num warmup instructions: {EVAL_WARMUP}")
    print(f" - Num simulation instructions: {EVAL_SIM}")
    print( " - Workloads used for testing:")
    for trace in test_traces:
        print("   -", os.path.basename(trace))

    eval_start = time.perf_counter()
    for trace in test_traces:
        with open("output.log", "w") as logfile:
            result = subprocess.run(["./build/predictor", "-c", "configs/default_config.json", 
                                     "-w", str(EVAL_WARMUP), 
                                     "-n", str(args.simulation_instructions), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)
        baseline = get_mpki()

        with open("output.log", "w") as logfile:
            result = subprocess.run(["./build/predictor", "-c", args.config, 
                                     "-w", str(args.warmup_instructions), 
                                     "-n", str(args.simulation_instructions), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)
        test_out = get_mpki()

        print(f" - {os.path.basename(trace)}:\tDefault Conf MPKI: {baseline}\tTest Conf MPKI: {test_out}")

    eval_end = time.perf_counter()

    print(f"\nNew config evaluation took {eval_end - eval_start} seconds\n")

