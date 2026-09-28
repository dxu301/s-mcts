from google import genai
from google.genai import types
import openai
from openai import OpenAI
import json
import copy
import re
import os
import requests
from collections import defaultdict
import itertools
import random
import numpy as np


class Solver:
    def __init__(self, solver_prompt, optimization_options, project_path, model, remove_component_index=None):

        self.model = model
        self.system_prompt = "You are a helpful assistant for travel planning."

        if ("gemini" in model):
            self.client = genai.Client(api_key=os.getenv('API_KEY'))
            thinking_config = types.ThinkingConfig(
                include_thoughts=True
            )
            self.generation_config = types.GenerateContentConfig(
                temperature=0.0, thinking_config=thinking_config
            )
        else:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        self.remove_component_index = remove_component_index

        with open(solver_prompt["planning"], "r") as f:
            self.generate_planning_prompt = f.read()

        with open(solver_prompt["planning_solution"], "r") as f:
            self.generate_planning_solution_prompt = f.read()

        with open(solver_prompt["combination"], "r") as f:
            self.generate_combination_prompt = f.read()

        with open(solver_prompt["solution"], "r") as f:
            self.generate_solution_prompt = f.read()

        with open(solver_prompt["deliver"], "r") as f:
            self.generate_deliver_prompt = f.read()
        # smcts_new_query
        with open(solver_prompt["smcts_new_query"], "r") as f:
            self.smcts_new_query_prompt = f.read()

        self.planning_optimization_prompt_list = []

        self.optimization_options = optimization_options

        self.group_parameter = None
        self.identify_keys_output = None
        self.set_cover_param = None

        for i in optimization_options:
            with open(f"scope/prompt/prompt_planning_optimization/planning_optimization_{i}_agent.txt", "r") as f:
                optimization_prompt = f.read()

            self.planning_optimization_prompt_list.append(optimization_prompt)

        with open(solver_prompt["input"], "r") as f:
            self.generate_input_prompt = f.read()

        with open(solver_prompt["smcts_selection"], "r") as f:
            self.smcts_selection_prompt = f.read()

        with open(solver_prompt["smcts_expansion"], "r") as f:
            self.smcts_expansion_prompt = f.read()

        with open(solver_prompt["smcts_policy"], "r") as f:
            self.smcts_policy_prompt = f.read()

        with open(solver_prompt["smcts_update_q"], "r") as f:
            self.smcts_update_q_prompt = f.read()

        with open(solver_prompt["smcts_summary"], "r") as f:
            self.smcts_summary_prompt = f.read()

        with open(solver_prompt["smcts_terminating_detect"], "r") as f:
            self.smcts_terminating_detect_prompt = f.read()

        self.combinations_func = None
        self.solutions_func = None
        self.deliver_func = None

        self.combinations_func_output = None
        self.solutions_func_output = None

        self.structured_output = None
        self.planning_output = None

        self.solution_cot = None
        self.solution_code = None

        if not os.path.exists(project_path):
            os.makedirs(project_path)

        self.project_path = project_path
        self.number_of_token_input = 0
        self.number_of_token_output = 0

    def call_api(self, prompt, get_tokens=False, t=1.0):
        if ("gemini" in self.model):
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self.generation_config
            )

            for part in response.parts:
                if part.thought:
                    continue
                else:
                    output = part.text

            usage = response.usage_metadata
            number_of_token_input = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
            thought_tokens = usage.thoughts_token_count
            number_of_token_output = output_tokens + thought_tokens

        else:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                temperature=t
            )
            output = chat_completion.choices[0].message.content
            number_of_token_input = chat_completion.usage.prompt_tokens
            number_of_token_output = chat_completion.usage.completion_tokens

        self.number_of_token_input += number_of_token_input
        self.number_of_token_output += number_of_token_output

        if (get_tokens):
            return output, number_of_token_input, number_of_token_output
        else:
            return output

    def check_sufficient(self, filtered_reference_data, minimum_requirement_item, key):
        minimum_requirement_item_dict = minimum_requirement_item[key]  # City/from_to : number

        if (key == "transport"):
            parameter = "from_to"
        else:
            parameter = "City"

        city_or_transport_count = {}
        for cat in minimum_requirement_item_dict:
            city_or_transport_count[cat] = 0

        for item in filtered_reference_data:
            city_or_transport_count[item[parameter]] += 1

        for cat in city_or_transport_count:
            min_num = int(minimum_requirement_item_dict[cat])
            if (city_or_transport_count[cat] < min_num):
                return False

        return True

    def create_solver(self, example_query, example_solution):

        # self.smcts_selection_prompt = self.smcts_selection_prompt.replace("<example_query>", few_shot_examples_no_answer)
        self.smcts_expansion_prompt = self.smcts_expansion_prompt.replace("<example_query>", example_query)
        self.smcts_policy_prompt = self.smcts_policy_prompt.replace("<example_query>", example_query)
        self.smcts_update_q_prompt = self.smcts_update_q_prompt.replace("<example_query>", example_query)
        self.smcts_terminating_detect_prompt = self.smcts_terminating_detect_prompt.replace("<example_query>", example_query)
        # self.smcts_expansion_prompt = self.smcts_expansion_prompt.replace("<example_reference>", json.dumps(reference_data, ensure_ascii=False, indent=4))
        # self.smcts_policy_prompt = self.smcts_policy_prompt.replace("<example_reference>", json.dumps(reference_data, ensure_ascii=False, indent=4))

        # structured_output, planning_output = self.create_planning(few_shot_examples_answer_only,
        #                                                           few_shot_examples_no_answer, options)
        # self.smcts_selection_prompt = self.smcts_selection_prompt.replace("<example_plan>", example_plan)
        self.smcts_expansion_prompt = self.smcts_expansion_prompt.replace("<example_plan>", example_solution)
        self.smcts_policy_prompt = self.smcts_policy_prompt.replace("<example_plan>", example_solution)
        self.smcts_update_q_prompt = self.smcts_update_q_prompt.replace("<example_plan>", example_solution)
        self.smcts_terminating_detect_prompt = self.smcts_terminating_detect_prompt.replace("<example_plan>", example_solution)

    def load_checkpoints(self):

        with open(f"{self.project_path}/self_improvement_planner_optimization_1.txt", "r") as f:
            identify_keys_output = f.read()

        match = re.search(
            r"<start_of_JSON>\s*(.*?)\s*<end_of_JSON>",
            identify_keys_output,
            re.DOTALL
        )
        if match:
            identify_keys_output = match.group(1).strip()

        self.identify_keys_output = identify_keys_output

        if (self.remove_component_index != 1):
            with open(f"{self.project_path}/self_improvement_planner_optimization_2.txt", "r") as f:
                analyst_output = f.read()

            self.set_cover_param = {
                category: values["set_cover"][0]
                for category, values in analyst_output.items()
                if "set_cover" in values and values["set_cover"]
            }

        self.group_parameter = {
            category: next(
                key for key, role in fields.items() if role == "combination"
            )
            for category, fields in identify_keys_output.items()
        }

        with open(f"{self.project_path}/self_improvement_planner_solution.txt", "r") as f:
            output = f.read()

        match = re.search(
            r"<start_of_structured_output>\s*(.*?)\s*<end_of_structured_output>",
            output,
            re.DOTALL
        )

        if match:
            match = match.group(1).strip()

        structured_solution_output = json.loads(match)

        f_name = f"{self.project_path}/self_improvement_planner.txt"

        if os.path.exists(f_name):
            with open(f_name, "r") as f:
                output = f.read()

                match = re.search(
                    r"<start_of_structured_output>\s*(.*?)\s*<end_of_structured_output>",
                    output,
                    re.DOTALL
                )
                if match:
                    match = match.group(1).strip()

                self.structured_output = json.loads(match)
                self.structured_output.update(structured_solution_output)

                input_agent_structured_output = self.structured_output.copy()

                if (self.remove_component_index == 0):
                    del input_agent_structured_output["parameters_description"]
                else:
                    del input_agent_structured_output["combinations_description"]
                    del input_agent_structured_output["constraints_description"]

                match = re.search(
                    r"<start_of_planning>\s*(.*?)\s*<end_of_planning>",
                    output,
                    re.DOTALL
                )
                if match:
                    match = match.group(1).strip()

                self.planning_output = json.loads(match)

                if (self.remove_component_index == 0):
                    del input_agent_structured_output["parameters"]["restaurants"]["set_cover"]
                    del input_agent_structured_output["parameters"]["accommodations"]["set_cover"]
                    del input_agent_structured_output["parameters"]["transport"]["set_cover"]
                else:
                    del input_agent_structured_output["constraints"]["restaurants"]["set_cover"]
                    del input_agent_structured_output["constraints"]["accommodations"]["set_cover"]
                    del input_agent_structured_output["constraints"]["transport"]["set_cover"]

                self.generate_input_prompt = self.generate_input_prompt.replace("<instructions>",
                                                                                self.planning_output["Input Agent"])
                # self.generate_input_prompt = self.generate_input_prompt.replace("<output_description>", output_description)
                self.generate_input_prompt = self.generate_input_prompt.replace("<example_output>", json.dumps(
                    input_agent_structured_output, ensure_ascii=False, indent=4))

        potential_checkpoints = ["combination", "solution", "deliver"]

        for cp in potential_checkpoints:

            if (cp == "solution"):
                f_name = f"{self.project_path}/self_improvement_{cp}.txt"
            else:
                f_name = f"{self.project_path}/self_improvement_{cp}.txt"

            if os.path.exists(f_name):
                with open(f_name, "r") as f:
                    output = f.read()
            else:
                continue

            if (cp == "combination"):
                match = re.search(
                    r"<start_of_code>\s*(.*?)\s*<end_of_code>",
                    output,
                    re.DOTALL
                )
                if match:
                    code_body = match.group(1).strip()

                local_scope = {}
                exec(code_body, local_scope)
                self.combinations_func = local_scope["combinations_func"]

                print("Combinations func is loaded")

            if (cp == "solution"):
                match = re.search(
                    r"<start_of_code>\s*(.*?)\s*<end_of_code>",
                    output,
                    re.DOTALL
                )
                if match:
                    code_body = match.group(1).strip()

                self.solution_code = code_body

                local_scope = {}
                exec(code_body, local_scope)
                self.solutions_func = local_scope["plan_func"]

                match = re.search(
                    r"<start_of_COT>\s*(.*?)\s*<end_of_COT>",
                    output,
                    re.DOTALL
                )
                if match:
                    self.solution_cot = match.group(1).strip()

                print("Solution func is loaded")

            if (cp == "deliver"):
                match = re.search(
                    r"<start_of_code>\s*(.*?)\s*<end_of_code>",
                    output,
                    re.DOTALL
                )
                if match:
                    code_body = match.group(1).strip()

                local_scope = {}
                exec(code_body, local_scope)
                self.deliver_func = local_scope["deliver_func"]

                print("Deliver func is loaded")

    def roll_out(self, new_node):
        current_plan = new_node["current_plan"]
        future_situation = ""
        t = 0
        while t <= 8:
            # terminal detection
            terminating_detect_prompt = self.smcts_terminating_detect_prompt.replace("<full_plan>", current_plan)
            detect_output = self.call_api(terminating_detect_prompt)
            detect_output = detect_output.replace("</", "<")
            detect_output = detect_output.replace("/>", ">")
            if detect_output.count("start_of_judgement") > 1:
                detect_output = "end_of_judgement".join(detect_output.rsplit("start_of_judgement", 1))
            elif detect_output.count("<start_of_judgement>") == 0:
                tmp_output = detect_output.splitlines()
                tmp_output[-3] = "<start_of_judgement>"
                detect_output = "\n".join(tmp_output)
            if detect_output.count("end_of_judgement") > 1:
                detect_output = detect_output.replace("end_of_judgement", "start_of_judgement", 1)
            elif detect_output.count("<end_of_judgement>") == 0:
                tmp_output = detect_output.splitlines()
                tmp_output[-1] = "<end_of_judgement>"
                detect_output = "\n".join(tmp_output)
            match = re.search(
                r"<start_of_judgement>\s*(.*?)\s*<end_of_judgement>",
                detect_output,
                re.DOTALL
            )
            if match:
                detect_output = match.group(1).strip()
            else:
                print(detect_output)
                assert False
            if "yes" in detect_output.lower():
                break
            policy_prompt = self.smcts_policy_prompt.replace("<current_plan>", current_plan)
            # print("==================Roll-out Policy Prompt=================")
            # print(policy_prompt)
            policy_output = self.call_api(policy_prompt)
            # print("==================Roll-out Policy Output=================")
            # print(policy_output)
            current_plan = current_plan + ",\n" + policy_output.strip(" ,\n")
            if future_situation == "":
                future_situation = policy_output.strip(" ,\n")
            else:
                future_situation = future_situation + ",\n" + policy_output.strip(" ,\n")
            t += 1

        if t >= 8:
            assert False

        new_node["future_situation"] = future_situation
        summary_prompt = self.smcts_summary_prompt.replace("<full_plan>", new_node["current_plan"] + ",\n" + new_node[
            "future_situation"])
        # print("==================Plan Summary Prompt=================")
        # print(summary_prompt)
        summary_output = self.call_api(summary_prompt)
        # print("==================Plan Summary Output=================")
        # print(summary_output)
        summary_output = summary_output.replace("</", "<")
        summary_output = summary_output.replace("/>", ">")
        if summary_output.count("start_of_judgement") > 1:
            summary_output = "end_of_judgement".join(summary_output.rsplit("start_of_judgement", 1))
        elif summary_output.count("<start_of_judgement>") == 0:
            tmp_output = summary_output.splitlines()
            tmp_output[-3] = "<start_of_judgement>"
            summary_output = "\n".join(tmp_output)
        if summary_output.count("end_of_judgement") > 1:
            summary_output = summary_output.replace("end_of_judgement", "start_of_judgement", 1)
        elif summary_output.count("<end_of_judgement>") == 0:
            tmp_output = summary_output.splitlines()
            tmp_output[-1] = "<end_of_judgement>"
            summary_output = "\n".join(tmp_output)
        match = re.search(
            r"<start_of_judgement>\s*(.*?)\s*<end_of_judgement>",
            summary_output,
            re.DOTALL
        )
        if match:
            summary_output = match.group(1).strip()
        else:
            print(summary_output)
            assert False
        new_node["future_summary"] = summary_output  # this is the summary of the full plan

        return new_node

    def solve(self, prompt_zero_shot):

        self.smcts_expansion_prompt = self.smcts_expansion_prompt.replace("<query>", prompt_zero_shot)
        self.smcts_policy_prompt = self.smcts_policy_prompt.replace("<query>", prompt_zero_shot)
        self.smcts_selection_prompt = self.smcts_selection_prompt.replace("<query>", prompt_zero_shot)
        self.smcts_update_q_prompt = self.smcts_update_q_prompt.replace("<query>", prompt_zero_shot)
        self.smcts_summary_prompt = self.smcts_summary_prompt.replace("<query>", prompt_zero_shot)
        self.smcts_terminating_detect_prompt = self.smcts_terminating_detect_prompt.replace("<query>", prompt_zero_shot)
        self.number_of_token_input = 0
        self.number_of_token_output = 0

        search_tree = [{"index": 0, "current_plan": "", "future_situation": "", "next_action_choices": {}}]
        root_not_updated = -1
        itr_n = 0
        while root_not_updated < 3:
            current_index = 0  # start from root node
            current_node = search_tree[current_index]
            # selection
            full_path = []
            while True:
                if len(current_node["next_action_choices"]) == 0:
                    full_path.append((current_index, None))
                    break

                selection_prompt = copy.deepcopy(self.smcts_selection_prompt)
                for i in range(len(current_node["next_action_choices"])):
                    visit_num_i = current_node["next_action_choices"][i]["visitation_number"]
                    child_i = search_tree[current_node["next_action_choices"][i]["child_index"]]
                    summary_i = child_i["future_summary"]
                    selection_prompt = selection_prompt.replace(f"<future_situation_{i + 1}>", summary_i)
                    selection_prompt = selection_prompt.replace(f"<visit_number_{i + 1}>", str(visit_num_i))
                    print(f"<visit_number_{i + 1}> --> {visit_num_i}")
                if len(current_node["next_action_choices"]) < 3:
                    for i in range(len(current_node["next_action_choices"]) + 1, 4):
                        selection_prompt = selection_prompt.replace(f"Action choice {i}:\n", "")
                        selection_prompt = selection_prompt.replace(f"Future situation: <future_situation_{i}>\n", "")
                        selection_prompt = selection_prompt.replace(f"Visitation number: <visit_number_{i}>", "")

                # print("==================Selection Prompt=================")
                # print(selection_prompt)
                selection_output = self.call_api(selection_prompt)
                print("==================Selection Output=================")
                print(selection_output)
                selection_output = selection_output.replace("</", "<")
                selection_output = selection_output.replace("/>", ">")
                if selection_output.count("start_of_answer") > 1:
                    selection_output = "end_of_answer".join(selection_output.rsplit("start_of_answer", 1))
                elif selection_output.count("<start_of_answer>") == 0:
                    tmp_output = selection_output.splitlines()
                    tmp_output[-3] = "<start_of_answer>"
                    selection_output = "\n".join(tmp_output)
                if selection_output.count("end_of_answer") > 1:
                    selection_output = selection_output.replace("end_of_answer", "start_of_answer", 1)
                elif selection_output.count("<end_of_answer>") == 0:
                    tmp_output = selection_output.splitlines()
                    tmp_output[-1] = "<end_of_answer>"
                    selection_output = "\n".join(tmp_output)
                match = re.search(
                    r"<start_of_answer>\s*(.*?)\s*<end_of_answer>",
                    selection_output,
                    re.DOTALL
                )
                if match:
                    selected_idx = match.group(1).strip()
                    selected_idx = int(selected_idx)
                    selected_idx -= 1  # important !!!
                    assert 0 <= selected_idx < 3
                else:
                    print(selection_output)
                    assert False

                print("==================Selected Action=================")
                print(current_node["next_action_choices"][selected_idx]["action"])

                full_path.append((current_index, selected_idx))
                next_act_ch = current_node["next_action_choices"]
                v_n = next_act_ch[selected_idx]["visitation_number"] + 1
                next_act_ch[selected_idx]["visitation_number"] = v_n

                current_index = next_act_ch[selected_idx]["child_index"]
                current_node = search_tree[current_index]

            # terminal detection
            terminating_detect_prompt = self.smcts_terminating_detect_prompt.replace("<full_plan>",
                                                                                     current_node["current_plan"])
            # print("==================Terminating Detection Prompt=================")
            # print(terminating_detect_prompt)
            detect_output = self.call_api(terminating_detect_prompt)
            # print("==================Terminating Detection Output=================")
            # print(detect_output)
            detect_output = detect_output.replace("</", "<")
            detect_output = detect_output.replace("/>", ">")
            if detect_output.count("start_of_judgement") > 1:
                detect_output = "end_of_judgement".join(detect_output.rsplit("start_of_judgement", 1))
            elif detect_output.count("<start_of_judgement>") == 0:
                tmp_output = detect_output.splitlines()
                tmp_output[-3] = "<start_of_judgement>"
                detect_output = "\n".join(tmp_output)
            if detect_output.count("end_of_judgement") > 1:
                detect_output = detect_output.replace("end_of_judgement", "start_of_judgement", 1)
            elif detect_output.count("<end_of_judgement>") == 0:
                tmp_output = detect_output.splitlines()
                tmp_output[-1] = "<end_of_judgement>"
                detect_output = "\n".join(tmp_output)
            match = re.search(
                r"<start_of_judgement>\s*(.*?)\s*<end_of_judgement>",
                detect_output,
                re.DOTALL
            )
            if match:
                detect_output = match.group(1).strip()
            else:
                print(detect_output)
                assert False
            if "no" in detect_output.lower():
                # expansion
                current_plan = current_node["current_plan"]
                expansion_prompt = self.smcts_expansion_prompt.replace("<current_plan>", current_plan)
                # print("==================Expansion Prompt=================")
                # print(expansion_prompt)
                expansion_output = self.call_api(expansion_prompt)
                print("==================Expansion Output=================")
                print(expansion_output)
                expansion_output = expansion_output.replace("</", "<")
                expansion_output = expansion_output.replace("/>", ">")
                if expansion_output.count("start_of_action") > 1:
                    expansion_output = "end_of_action".join(expansion_output.rsplit("start_of_action", 1))
                if expansion_output.count("end_of_action") > 1:
                    expansion_output = expansion_output.replace("end_of_action", "start_of_action", 1)
                elif expansion_output.count("<end_of_action>") == 0:
                    tmp_output = expansion_output.splitlines()
                    tmp_output[-1] = "<end_of_action>"
                    expansion_output = "\n".join(tmp_output)
                match = re.search(
                    r"<start_of_action>\s*(.*?)\s*<end_of_action>",
                    expansion_output,
                    re.DOTALL
                )
                if match:
                    expansion_output = match.group(1).strip()
                else:
                    print(expansion_output)
                    assert False
                if len(expansion_output) > 0:
                    new_actions = expansion_output.split("---")
                    assert len(new_actions) <= 3
                    full_plan_list = []
                    summary_list = []
                    for ch_idx, new_act in enumerate(new_actions):
                        # create new child node
                        new_act = new_act.strip(" ,\n")
                        if len(search_tree) > 1:
                            new_node = {"index": len(search_tree), "current_plan": current_plan + ",\n" + new_act,
                                        "future_situation": "", "next_action_choices": {}}
                        else:
                            new_node = {"index": len(search_tree), "current_plan": new_act, "future_situation": "",
                                        "next_action_choices": {}}

                        # roll-out for the new node, "future_situation" is initialized here
                        new_node = self.roll_out(new_node)

                        full_plan_list.append(
                            new_node["current_plan"].strip(", ") + ",\n" + new_node["future_situation"].strip(", "))
                        summary_list.append(new_node["future_summary"])
                        print(f"------------------------Full Plan of New Node {ch_idx + 1}------------------------")
                        print(full_plan_list[-1])
                        print(new_node["future_summary"])
                        new_act_dict = {"action": new_act, "visitation_number": 0, "child_index": new_node["index"]}
                        current_node["next_action_choices"][ch_idx] = new_act_dict
                        search_tree.append(new_node)

                    # get the future situation of the node just expanded
                    future_score = [0 for _ in range(len(new_actions))]
                    for i in range(len(new_actions)):
                        for j in range(len(new_actions)):
                            if i < j:
                                plan_A = full_plan_list[i]
                                plan_B = full_plan_list[j]
                                # update_prompt = self.smcts_update_q_prompt.replace("<plan_A>", plan_A)
                                update_prompt = self.smcts_update_q_prompt.replace("<summary_A>", summary_list[i])
                                # update_prompt = update_prompt.replace("<plan_B>", plan_B)
                                update_prompt = update_prompt.replace("<summary_B>", summary_list[j])
                                # print("==================Update Prompt=================")
                                # print(update_prompt)
                                update_output = self.call_api(update_prompt)
                                if "A" in update_output:
                                    future_score[i] += 1
                                else:
                                    future_score[j] += 1
                    best_future_idx = np.argmax(future_score).item()
                    best_child_idx = current_node["next_action_choices"][best_future_idx]["child_index"]
                    best_child_node = search_tree[best_child_idx]
                    best_action = current_node["next_action_choices"][best_future_idx]["action"]
                    current_node["future_situation"] = best_action + ",\n" + best_child_node["future_situation"]
                    current_node["future_summary"] = best_child_node["future_summary"]
                    # print(current_node["future_situation"])

            # back-propagation (update)
            if current_node["current_plan"] != "":
                full_plan_A = current_node["current_plan"] + ",\n" + current_node["future_situation"]
            else:
                full_plan_A = current_node["future_situation"]
            summary_A = current_node["future_summary"]
            updated_or_not = False
            for node_idx, act_idx in reversed(full_path[:-1]):
                current_node = search_tree[node_idx]
                if current_node["current_plan"] != "":
                    full_plan_B = current_node["current_plan"] + ",\n" + current_node["future_situation"]
                else:
                    full_plan_B = current_node["future_situation"]
                # update_prompt = self.smcts_update_q_prompt.replace("<plan_A>", full_plan_A)
                update_prompt = self.smcts_update_q_prompt.replace("<summary_A>", summary_A)
                # update_prompt = update_prompt.replace("<plan_B>", full_plan_B)
                update_prompt = update_prompt.replace("<summary_B>", current_node["future_summary"])
                # print("==================Update Prompt=================")
                # print(update_prompt)
                update_output = self.call_api(update_prompt)
                if "A" in update_output:
                    if node_idx == 0:
                        updated_or_not = True
                    next_node_idx = current_node["next_action_choices"][act_idx]["child_index"]
                    next_act = current_node["next_action_choices"][act_idx]["action"]
                    new_future = next_act + ",\n" + search_tree[next_node_idx]["future_situation"]
                    current_node["future_situation"] = new_future
                    current_node["future_summary"] = search_tree[next_node_idx]["future_summary"]

            if updated_or_not:
                root_not_updated = 0
            else:
                root_not_updated += 1

            itr_n += 1
            if itr_n > 20:
                assert False

        # print("Input Token: {}".format(self.number_of_token_input))
        # print("Output Token: {}".format(self.number_of_token_output))
        return search_tree[0]["future_situation"], self.number_of_token_input, self.number_of_token_output