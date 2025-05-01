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
                    help="Input config JSON file")
args = parser.parse_args()

trace_dir="./traces/"
trace_ext = ".champsim.trace.gz"
testing_workloads = [
        "nodeapp-nodeapp", 
        "dacapo-spring", 
        "delta.507252", 
        "renaissance-finagle-chirper"
        ]

with open("output/baseline_mpki.json", "r") as f:
    initial_mpki = json.load(f)

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
            result = subprocess.run(["./build/predictor", "-c", args.config, 
                                     "-w", str(EVAL_WARMUP), 
                                     "-n", str(EVAL_SIM), trace],
                                     stdout=logfile, stderr=subprocess.STDOUT, text=True)
        test_out = get_mpki()
        baseline = initial_mpki[trace]

        print(f" - {os.path.basename(trace)}:\tDefault Conf MPKI: {baseline}\tTest Conf MPKI: {test_out}")

    eval_end = time.perf_counter()

    print(f"\nNew config evaluation took {eval_end - eval_start} seconds\n")

