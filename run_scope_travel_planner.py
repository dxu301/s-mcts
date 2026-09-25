import json
import time
import os
import re
from scope.workflow_with_reference_data import Solver
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
import argparse
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from scope.utils import read_jsonl_to_list, restructure_data

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="gpt-5.6-sol", type=str)
parser.add_argument("--load_checkpoint", default=False, type=str)
parser.add_argument("--remove_component_index", default=None, type=int)

args = parser.parse_args()

model = args.model
remove_component_index = args.remove_component_index

if(remove_component_index==None):
    output_file = f"travel_planner_{model}/travel_planner_scope.json"
else:
    remove_component_dict = {
        0: "problem_formalisation",
        1: "problem_optimisation",
        2: "code_refinement"
    }

    remove_component = remove_component_dict[remove_component_index]

    output_file = f"otravel_planner_{model}/travel_planner_scope_{remove_component}.json"

solver_prompt = {
    "planning": "scope/prompt/prompt_scope_travel_planner/planning_agent.txt",
    "planning_solution": "scope/prompt/prompt_scope_travel_planner/solution_agent.txt",
    "combination": "scope/prompt/prompt_scope_travel_planner/combination_function_generator_agent.txt",
    "solution": "scope/prompt/prompt_scope_travel_planner/solution_function_generator_agent.txt",
    "deliver": "scope/prompt/prompt_scope_travel_planner/deliver_function_generator_agent.txt",
    "input": "scope/prompt/prompt_scope_travel_planner/input_agent.txt",
    "smcts_policy": "scope/prompt/prompt_scope_travel_planner/smcts_policy.txt",
    "smcts_selection": "scope/prompt/prompt_scope_travel_planner/smcts_selection.txt",
    "smcts_expansion": "scope/prompt/prompt_scope_travel_planner/smcts_expansion.txt",
    "smcts_update_q": "scope/prompt/prompt_scope_travel_planner/smcts_update_q.txt",
    "smcts_terminating_detect": "scope/prompt/prompt_scope_travel_planner/smcts_terminating_detect.txt",
    "smcts_summary": "scope/prompt/prompt_scope_travel_planner/smcts_summary.txt",
    "smcts_new_query": "scope/prompt/prompt_scope_travel_planner/smcts_generate_conflicting_query2.txt",
    "combination_reflection": "scope/prompt/prompt_scope_travel_planner/combination_function_generator_reflection_agent.txt",
    "solution_reflection": "scope/prompt/prompt_scope_travel_planner/solution_function_generator_reflection_agent.txt",
    "deliverer_reflection": "scope/prompt/prompt_scope_travel_planner/deliverer_function_generator_reflection_agent.txt"
}

if(remove_component_index==0):
    solver_prompt["planning"] = "prompt/prompt_scope_travel_planner/planning_agent_0.txt"
    solver_prompt["combination"] = "prompt/prompt_scope_travel_planner/combination_function_generator_agent_0.txt"
    solver_prompt["solution"] = "prompt/prompt_scope_travel_planner/solution_function_generator_agent_0.txt"


# The Optimization Component for SCOPE with reference data is as follows :
# 1. Extracting options for inference (this is compulsory)
# 2. Design the format for Combinations Agent (this is compulsory)
# 3. Idenitfy which item to be filtered, sorted and pruned (this is optional)

# Prompt 5 is required for 1,2. Prompt 6 is required for 3.

if(remove_component_index==1):
    optimization_options = [5]
else:
    optimization_options = [5,6]

if(args.load_checkpoint!=False):
    solver = Solver(solver_prompt, optimization_options, args.load_checkpoint, model, remove_component_index)
else:
    solver = Solver(solver_prompt, optimization_options, f"travel_planner_{model}", model, remove_component_index)

with open(f"scope/prompt/example_data.json", "r") as f:
    example_data = json.load(f)

example_query = example_data["travel_planner"]["query"]
example_solution = example_data["travel_planner"]["solution"]


if os.path.exists(output_file):
    with open(output_file, "r") as f:
        generated_data = json.load(f)
else:
    generated_data = {}

print(f"Dataset : Travel Planner | Model : {model}")

with open("scope/prompt/prompt_scope_travel_planner/structure.txt", "r") as f:
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
output_path = f"travel_planner_{model}/travel_planner_scope.json"
if os.path.exists(output_path):
    with open(output_path) as f:
        generated_data = json.load(f) 
else:
    generated_data = {}
    with open(output_path,"w") as f:
        json.dump({}, f) 

# generate new queries with multi-level constraints
for _ in tqdm(range(15)):
    number = np.random.randint(1, 1000)
    query_data = query_data_list[number - 1]
    reference_information = test_reference_informations[number - 1]
    reference_information = restructure_data(reference_information)

    new_query = solver.generate_new_query(query_data["query"], reference_information)
    new_line = str(number) + ": " + new_query + "\n"

    with open("new_queries_2-level_constraints.txt", "a") as file:
        file.write(new_line)

assert False

for number in tqdm(numbers[:]):

    if(str(number) in generated_data):
        continue

    if number < 530:
        continue
        
    data = {}
    
    query_data = query_data_list[number-1]
    reference_information = test_reference_informations[number-1]
    reference_information = restructure_data(reference_information)

    start_time = time.time()
    print("----------------------Query---------------------")
    print(query_data["query"])
    new_query = solver.generate_new_query(query_data["query"], reference_information)
    structured_output, final_answer, combination_size, number_of_token_input, number_of_token_output = solver.solve(query_data["query"], reference_information)

    data["prompt_0shot"] = query_data["query"]
    data["structured_output"] = structured_output
    data["number_of_input_token"] = number_of_token_input
    data["number_of_output_token"] = number_of_token_output
    data["plan"] = final_answer

    data["combination_size"] = combination_size
    data["time_taken"] = time.time() - start_time

    if(final_answer!=None):
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
    else:
        json_output = None

    data["structured_plan"] = json_output

    generated_data[number] = data

    with open(output_path, "w") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

