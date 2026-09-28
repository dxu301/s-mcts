import json
import time
import os
import argparse
from workflow import Solver

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="gpt-5.6-luna", type=str)
parser.add_argument("--dataset", default="trip_planning", type=str)
parser.add_argument("--load_checkpoint", default=None, type=str)
parser.add_argument("--remove_component_index", default=None, type=int)

args = parser.parse_args()

model = args.model
dataset = args.dataset
remove_component_index = args.remove_component_index


output_file = f"output_full_{model}/{dataset}_smcts_{remove_component}.json"

if(dataset not in ["trip_planning", "meeting_planning"]):
    raise Exception("Dataset not found")

solver_prompt = {
    "smcts_policy": "prompt/prompt_trip_planning/smcts_policy.txt",
    "smcts_selection": "prompt/prompt_trip_planning/smcts_selection.txt",
    "smcts_expansion": "prompt/prompt_trip_planning/smcts_expansion.txt",
    "smcts_update_q": "prompt/prompt_trip_planning/smcts_update_q.txt",
    "smcts_terminating_detect": "prompt/prompt_trip_planning/smcts_terminating_detect.txt",
    "smcts_summary": "prompt/prompt_trip_planning/smcts_summary.txt",
    "smcts_new_query": "prompt/prompt_trip_planning/smcts_generate_conflicting_query2.txt",
}

with open(f"prompt/example_data.json", "r") as f:
    example_data = json.load(f)

example_query = example_data[dataset]["query"]
example_solution = example_data[dataset]["solution"]

with open(f"data/{dataset}.json", "r") as f:
    test_data = json.load(f)

# we recommend using Optimization Agent 1,2,3 for Trip Planning and 4 for Meeting Planning
if(dataset=="trip_planning"):
    optimization_options = [1,2,3]
else:
    optimization_options = [4]

if(args.load_checkpoint!=None):
    solver = Solver(solver_prompt, optimization_options, args.load_checkpoint, model, remove_component_index)
else:
    solver = Solver(solver_prompt, optimization_options, f"{dataset}_{model}_{remove_component_index}", model, remove_component_index)


if(args.load_checkpoint):
    solver.load_checkpoints(example_query)
else:
    solver.create_solver(example_query, example_solution)


if os.path.exists(output_file):
    with open(output_file, "r") as f:
        generated_data = json.load(f)
else:
    generated_data = {}


print(f"Dataset : {dataset}| Model : {model}")
    

for d in test_data:

    index = int(d.split("_")[-1])
        
    if(d in generated_data):
        continue

    data = test_data[d]

    start_time = time.time()
    final_answer, number_of_token_input, number_of_token_output = solver.solve(data["prompt_0shot"])

    data["prompt_0shot"] = query_data["query"]
    data["number_of_input_token"] = number_of_token_input
    data["number_of_output_token"] = number_of_token_output
    data["plan"] = final_answer

    generated_data[d] = data

    with open(output_file, "w") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)
    
