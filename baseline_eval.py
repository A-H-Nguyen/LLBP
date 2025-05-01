import argparse
import json
import random
import copy
import re
import subprocess
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-w", "--warmup_instructions", type=int, required=True, 
                    help="Number of warmup instructions")
parser.add_argument("-n", "--simulation_instructions", type=int, required=True, 
                    help="Number of instructions of the region of interest (ROI)")
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


if __name__ == "__main__":
    initial_mpki = {}

    train_traces = [trace_dir + workload + trace_ext for workload in training_workloads]
    print("*** Initial MPKI values ***\n")
    print("Training:")
    print(f" - Num warmup instructions: {args.warmup_instructions}")
    print(f" - Num simulation instructions: {args.simulation_instructions}")
    print( " - Workloads used for training:")
    for trace in train_traces:
        print("   -", os.path.basename(trace))

    for trace in train_traces:
        with open("output.log", "w") as logfile:
            result = subprocess.run(["./build/predictor", "-c", "configs/default_config.json", 
                                     "-w", str(args.warmup_instructions), 
                                     "-n", str(args.simulation_instructions), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)
        new_mpki = get_mpki()
        initial_mpki[trace] = new_mpki 
        print(f" - {os.path.basename(trace)}:\t{new_mpki}")

    print("=" * 55, "\n")

    EVAL_WARMUP = 100000000
    EVAL_SIM    = 200000000

    test_traces = [trace_dir + workload + trace_ext for workload in testing_workloads]

    print("Testing:")
    print(f" - Num warmup instructions: {EVAL_WARMUP}")
    print(f" - Num simulation instructions: {EVAL_SIM}")
    print( " - Workloads used for testing:")
    for trace in test_traces:
        print("   -", os.path.basename(trace))
    for trace in test_traces:
        with open("output.log", "w") as logfile:
            result = subprocess.run(["./build/predictor", "-c", args.config, 
                                     "-w", str(EVAL_WARMUP ), 
                                     "-n", str(EVAL_SIM    ), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)
        new_mpki = get_mpki()
        initial_mpki[trace] = new_mpki 
        print(f" - {os.path.basename(trace)}:\t{new_mpki}")

    with open("output/baseline_mpki.json", "w") as f:
        json.dump(initial_mpki, f)
