import json
import time
import os
import re
from workflow_with_reference_data import Solver
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
import argparse
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from utils import read_jsonl_to_list, restructure_data

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="gpt-5.6-sol", type=str)
parser.add_argument("--load_checkpoint", default=False, type=str)
parser.add_argument("--remove_component_index", default=None, type=int)

args = parser.parse_args()

model = args.model
remove_component_index = args.remove_component_index

if(remove_component_index==None):
    output_file = f"travel_planner_{model}/travel_planner_smcts.json"
else:
    remove_component_dict = {
        0: "problem_formalisation",
        1: "problem_optimisation",
        2: "code_refinement"
    }

    remove_component = remove_component_dict[remove_component_index]

    output_file = f"travel_planner_{model}/travel_planner_smcts_{remove_component}.json"

solver_prompt = {
    "smcts_policy": "prompt/prompt_travel_planner/smcts_policy.txt",
    "smcts_selection": "prompt/prompt_travel_planner/smcts_selection.txt",
    "smcts_expansion": "prompt/prompt_travel_planner/smcts_expansion.txt",
    "smcts_update_q": "prompt/prompt_travel_planner/smcts_update_q.txt",
    "smcts_terminating_detect": "prompt/prompt_travel_planner/smcts_terminating_detect.txt",
    "smcts_summary": "prompt/prompt_travel_planner/smcts_summary.txt",
    "smcts_new_query": "prompt/prompt_travel_planner/smcts_generate_conflicting_query2.txt",
}


if(args.load_checkpoint!=False):
    solver = Solver(solver_prompt, optimization_options, args.load_checkpoint, model, remove_component_index)
else:
    solver = Solver(solver_prompt, optimization_options, f"travel_planner_{model}", model, remove_component_index)

with open(f"prompt/example_data.json", "r") as f:
    example_data = json.load(f)

example_query = example_data["travel_planner"]["query"]
example_solution = example_data["travel_planner"]["solution"]


if os.path.exists(output_file):
    with open(output_file, "r") as f:
        generated_data = json.load(f)
else:
    generated_data = {}

print(f"Dataset : Travel Planner | Model : {model}")

with open("prompt/prompt_travel_planner/structure.txt", "r") as f:
    conversion_prompt_template = f.read()

# we recommend using OpenAI GPT-4o for the conversion format of solution due to its strong instruction following capabilities
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

example_reference_data = example_data["travel_planner"]["reference_data"]

# For convenience of experiment, we restructure the reference data for better retrieval of information
# All baselines receive the same restructured data for fairness

example_reference_data = restructure_data(example_reference_data)


if(args.load_checkpoint):
    solver.load_checkpoints()
else:
    solver.create_solver(example_query, example_solution, example_reference_data)

query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']
numbers = [i for i in range(1,len(query_data_list)+1)]

test_reference_informations = read_jsonl_to_list(f"test_ref_info.jsonl")
output_path = f"travel_planner_{model}/travel_planner_smcts.json"
if os.path.exists(output_path):
    with open(output_path) as f:
        generated_data = json.load(f) 
else:
    generated_data = {}
    with open(output_path,"w") as f:
        json.dump({}, f) 

# generate new queries with multi-level constraints
# for _ in tqdm(range(258)):
#     number = np.random.randint(1, 1000)
#     query_data = query_data_list[number - 1]
#     reference_information = test_reference_informations[number - 1]
#     reference_information = restructure_data(reference_information)

#     new_query = solver.generate_new_query(query_data["query"], reference_information)
#     new_line = str(number) + ": " + new_query + "\n"

#     with open("new_queries_2-level_constraints.txt", "a") as file:
#         file.write(new_line)

for number in tqdm(numbers[:]):

    if(str(number) in generated_data):
        continue

    data = {}
    
    query_data = query_data_list[number-1]
    reference_information = test_reference_informations[number-1]
    reference_information = restructure_data(reference_information)

    start_time = time.time()
    # print("----------------------Query---------------------")
    # print(query_data["query"])
    # new_query = solver.generate_new_query(query_data["query"], reference_information)
    final_answer, number_of_token_input, number_of_token_output = solver.solve(query_data["query"], reference_information)

    data["prompt_0shot"] = query_data["query"]
    data["number_of_input_token"] = number_of_token_input
    data["number_of_output_token"] = number_of_token_output
    data["plan"] = final_answer

    data["time_taken"] = time.time() - start_time

    json_output = None
    if final_answer != None:
        conversion_prompt = conversion_prompt_template.replace("<text>", final_answer)
        convert_fail = True
        while(convert_fail):
            conversion_output = client.responses.create(
                model="gpt-4o",
                instructions=None,
                input=conversion_prompt, temperature=0.0
            ).output_text

            try:
                conversion_output = conversion_output.strip()
                match = re.search(
                    r"<start_of_JSON>\s*(.*?)\s*<end_of_JSON>", 
                    conversion_output, 
                    re.DOTALL
                ) 
                json_output = match.group(1).strip()
                json_output = json.loads(json_output)
                convert_fail = False
            except:
                convert_fail = True

    data["structured_plan"] = json_output

    generated_data[number] = data

    with open(output_path, "w") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

