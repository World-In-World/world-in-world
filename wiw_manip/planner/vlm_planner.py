import os.path as osp
import traceback
import numpy as np
import cv2
import json
import ast
import random
import time
import os
import datetime as _dt
from wiw_manip.envs.eb_man_utils import ROTATION_RESOLUTION, VOXEL_SIZE
from wiw_manip.planner.utils.remote_model import RemoteModel
from wiw_manip.planner.utils.custom_model import CustomModel
from wiw_manip.planner.base_planner import BasePlanner
from wiw_manip.planner.utils.planner_utils import local_image_to_data_url, template_manip, template_lang_manip
from wiw_manip.main import logger
from json_repair import repair_json
from typing import List, Tuple, Optional, Any, Dict
from wiw_manip.planner.utils.saver import (
    format_chat_dialog,
)

VISUAL_ICL_EXAMPLES_PATH = "wiw_manip/evaluator/config/visual_icl_examples/eb_manipulation"
VISUAL_ICL_EXAMPLE_CATEGORY = {
    "pick": "pick_cube_shape",
    "place": "place_into_shape_sorter_color",
    "stack": "stack_cubes_color",
    "wipe": "wipe_table_direction"
}

class VLMPlanner(BasePlanner):

    def __init__(
        self,
        model_name,
        model_type,
        system_prompt,
        examples,
        n_shot=0,
        obs_key="front_rgb",
        chat_history=False,
        language_only=False,
        multiview=False,
        multistep=False,
        visual_icl=False,
        tp=1,
        executed_action_per_step=1,
        action_length=7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_type = model_type
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.action_length = action_length
        self.chat_history = chat_history # whether to include all the chat history for prompting
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            vlm_args = kwargs.pop("vlm_args", {})
            self.model = RemoteModel(
                model_name,
                model_type,
                language_only,
                tp=tp,
                task_type="manip",
                **vlm_args,
            )
            self.vlm_args = vlm_args

        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.multi_view = multiview
        self.multi_step_image = multistep
        self.visual_icl = visual_icl
        self.executed_action_per_step = executed_action_per_step

    def _build_general_examples_prompt(self, task_variation, system_prompt, give_examples=True):
        if self.n_shot >= 1 and give_examples:
            appended_examples = self._build_appended_examples(task_variation)
            general_prompt = self._build_general_prompt(
                system_prompt, appended_examples
            )
        else:
            appended_examples = ''
            general_prompt = self._build_general_prompt(
                system_prompt, appended_examples
            )
        return general_prompt

    def process_prompt(self, system_prompt, user_instruction, avg_obj_coord, task_variation, prev_act_feedback=[], give_examples=True):
        if len(prev_act_feedback) == 0:
            general_prompt = self._build_general_examples_prompt(task_variation, system_prompt, give_examples)
            task_prompt = self._build_task_prompt(user_instruction, avg_obj_coord)
        elif self.chat_history:
            general_prompt = f'The human instruction is: {user_instruction}.'
            general_prompt += '\n\n The gripper action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                general_prompt += (
                    "\n Step {}, the output action **{}**, env feedback: {}".format(
                        i, action_feedback[0], action_feedback[1]
                    )
                )
            task_prompt = self._build_task_prompt(
                user_instruction, avg_obj_coord
            )
        else:
            general_prompt = self._build_general_examples_prompt(task_variation, system_prompt, give_examples)
            task_prompt = self._build_task_prompt(
                user_instruction, avg_obj_coord
            )
        return general_prompt, task_prompt

    # ---------------- Prompt Construction Helpers ----------------
    def _build_task_prompt(self, user_instruction, avg_obj_coord):
        if self.chat_history:
            task_prompt = (
                f"\n\n## Considering the above interaction history and the current image state, "
                f"to achieve the human instruction: '{user_instruction}', you are supposed to output in json. "
                f"You need to describe current visual state from the image, summarize interaction history and environment feedback "
                f"and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. "
                f"At the end, output the executable plan with the 7-dimsension action."
            )
        else:
            task_prompt = (
                f"\n## Now you are supposed to follow the above examples to generate a sequence of "
                f"discrete gripper actions that completes the below human instruction."
                f"\n\n**Human Instruction**: {user_instruction}.\n\n**Input**: {avg_obj_coord}\n\n**Output**: gripper actions"
            )
        return task_prompt

    def _build_general_prompt(self, template, appended_examples):
        general_prompt = template.format(
            VOXEL_SIZE,
            VOXEL_SIZE,
            int(360 / ROTATION_RESOLUTION),
            ROTATION_RESOLUTION,
            """Below are some examples to guide you in completing the task.\n\n""" +
            appended_examples if appended_examples else "",
        )
        return general_prompt

    def _build_appended_examples(self, task_variation):
        key = "_".join(task_variation.split("_")[:-1])  # Get the string before the last "_"
        return "\n".join(
            [
                f"Example {i}: \n{x}"
                for i, x in enumerate(
                    self.examples[key][: self.n_shot]
                )
            ]
        )
    # ---------------- Prompt Construction Helpers ----------------

    def process_prompt_visual_icl(self, user_instruction, avg_obj_coord, prev_act_feedback=[]):
        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            task_prompt = f"## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
        elif self.chat_history:
            general_prompt = f'The human instruction is: {user_instruction}.'
            general_prompt += '\n\n The gripper action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                general_prompt += '\n Step {}, the output action **{}**, env feedback: {}'.format(i, action_feedback[0], action_feedback[1])
            task_prompt = f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with the 7-dimsension action.'''
        else:
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            task_prompt = f"## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
            for i, action_feedback in enumerate(prev_act_feedback):
                task_prompt += f"{action_feedback}, "
        return general_prompt, task_prompt

    def get_message(self, images, prompt, task_prompt, last_act=None, messages=[]):
        if self.language_only and not self.visual_icl:
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + task_prompt}],
                }
            ]
        else:
            if self.multi_step_image:
                current_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}],
                    }
                ]

                # use the last three imags as multi-step images
                if len(images) >= 3:
                    multi_step_images = images[-3:-1]
                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": "You are given the scene observations from the last two steps:",
                        }
                    )
                    for image in multi_step_images:
                        if type(image) == str:
                            image_path = image
                        else:
                            image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                            cv2.imwrite(image_path, image)
                        data_url = local_image_to_data_url(image_path=image_path)
                        current_message[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                }
                            }
                        )

                    # add the current task prompt and input image
                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": task_prompt,
                        }
                    )

                    # add the current step image
                    current_step_image = images[-1]
                    if type(current_step_image) == str:
                        image_path = current_step_image
                    else:
                        raise ValueError("images should be path.")
                        image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                        cv2.imwrite(image_path, current_step_image)
                    data_url = local_image_to_data_url(image_path=image_path)
                    current_message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }
                    )
                else:
                    full_prompt = prompt + task_prompt
                    current_message = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": full_prompt}],
                        }
                    ]

                    for image in images:
                        if type(image) == str:
                            image_path = image
                        else:
                            raise ValueError("images should be path.")
                            image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                            cv2.imwrite(image_path, image)

                        data_url = local_image_to_data_url(image_path=image_path)
                        current_message[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                }
                            }
                        )

            else:
                full_prompt = prompt + task_prompt
                current_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt}],
                    }
                ]

                for image in images:
                    if type(image) == str:
                        image_path = image
                    else:
                        raise ValueError("images should be path.")
                        image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                        cv2.imwrite(image_path, image)

                    data_url = local_image_to_data_url(image_path=image_path)
                    current_message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }
                    )
            if last_act is not None:
                current_message[0]["content"].append(
                    {
                        "type": "text",
                        "text": last_act,
                    }
                )
            return current_message

    def get_message_visual_icl(self, images, first_prompt, task_prompt, task_variation, messages=[]):
        current_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_prompt}
                ],
            }
        ]
        visual_task_variation = VISUAL_ICL_EXAMPLE_CATEGORY[task_variation.split('_')[0]]
        task_specific_image_example_path = osp.join(VISUAL_ICL_EXAMPLES_PATH, visual_task_variation)
        icl_text_examples = self.examples[task_variation]
        stop_idx = 2
        for example_idx, example in enumerate(icl_text_examples):
            if example_idx >= stop_idx:
                break
            current_image_example_path = osp.join(task_specific_image_example_path, f"episode_{example_idx+1}_step_0_front_rgb_annotated.png")
            example = "Example {}:\n{}".format(example_idx+1, example)
            data_url = local_image_to_data_url(image_path=current_image_example_path)

            # Add the example image and the corresponding text to the message
            current_message[0]["content"].append(
                {
                    "type": "text",
                    "text": example,
                }
            )
            current_message[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "name": current_image_example_path,
                    }
                }
            )
        # add the task prompt
        current_message[0]["content"].append(
            {
                "type": "text",
                "text": task_prompt,
            }
        )

        for image in images:
            if type(image) == str:
                image_path = image
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)

            data_url = local_image_to_data_url(image_path=image_path)
            current_message[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "name": image_path,
                    }
                }
            )
        return current_message

    def json_to_action(self, output_text: str) -> Tuple[List[List[float]], Optional[Dict[str, Any]]]:
        """Parse *output_text* into a list of executable actions.

        The method is designed to be fault‑tolerant:
        * If the JSON is malformed or the structure is unexpected, it falls back to a single
          random action so the planner can keep running.
        * ``self.output_json_error`` is incremented on every recoverable error and set to
          ``-1`` when the executable plan is intentionally empty (signals the end of an
          episode).

        Parameters
        ----------
        output_text
            A (possibly malformed) JSON string that should contain an *executable_plan*.

        Returns
        -------
        actions
            A list of actions where each action is a list of float/int values understood by
            the low‑level controller.
        json_object
            The parsed JSON object on success, or ``None`` when the method had to fall back
            to a random action.
        """
        # * 1. try to parse the output text as JSON
        json_object = repair_json(output_text, return_objects=True)

        # * 2. try to extract the executable plan from the JSON object
        executable_plan = self._extract_plan(json_object)

        # * 3. ensure the executable plan is a list
        executable_plan = self._ensure_list(executable_plan)

        if len(executable_plan) == 0:
            raise ValueError("The parsered executable plan is empty, which is not allowed.")
        else:
            # * 4. normalize each step in the executable plan to a list of floats
            actions = [self._normalize_action(step) for step in executable_plan]
            return actions, json_object

    # ----------------------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------------------
    def _random_action(self) -> List[float]:
        """Generate a single random action compatible with the controller."""
        arm = [random.randint(0, VOXEL_SIZE) for _ in range(3)]
        rot = [random.randint(0, (360 // ROTATION_RESOLUTION) - 1) for _ in range(3)]
        gripper = [1.0]  # Always open
        return arm + rot + gripper

    def _bump_error(self) -> None:
        self.output_json_error += 1

    @staticmethod
    def _extract_plan(json_object: Dict[str, Any]) -> Optional[Any]:
        """Return the *executable_plan* entry if present; otherwise *None*."""
        if "executable_plan" in json_object:
            plan = json_object["executable_plan"]
        elif "properties" in json_object and "executable_plan" in json_object["properties"]:
            plan = json_object["properties"]["executable_plan"]
        else:
            raise ValueError("No executable_plan found in JSON")
        return plan

    @staticmethod
    def _ensure_list(plan: Any) -> Optional[List[Any]]:
        """Coerce *plan* into a Python list, handling stringified literals."""
        if isinstance(plan, str):
            plan = ast.literal_eval(plan)
        # plan is list tuple or None
        assert isinstance(plan, (list, tuple)), f"Expected a list or tuple or None, got {type(plan)}: {plan}"
        return plan

    def _normalize_action(self, raw_step: Any) -> List[float]:
        """Extract and return the *action* list from *raw_step* in its many possible forms."""
        # Unwrap tuples that come from JSON arrays interpreted as tuples
        if isinstance(raw_step, tuple):
            raw_step = list(raw_step)

        # Case 1 – direct dict with key "action"
        if isinstance(raw_step, dict) and "action" in raw_step:
            action = raw_step["action"]
        # Case 2 – already a flat list like [x, y, z, ...]
        elif isinstance(raw_step, list) and raw_step and isinstance(raw_step[0], int):
            action = raw_step
        # Case 3 – nested once more: [{"action": [...]}, ...]
        elif (
            isinstance(raw_step, list)
            and raw_step
            and isinstance(raw_step[0], dict)
            and "action" in raw_step[0]
        ):
            action = raw_step[0]["action"]
        else:
            action = raw_step  # Fallback – trust whatever we got

        # Convert string representation of list to actual list if needed
        if isinstance(action, str):
            action = ast.literal_eval(action)

        return action

    def reset(self):
        # at the beginning of the episode
        super().reset()
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def act(self, observation, user_instruction, avg_obj_coord, task_variation, query_num=1, temperature=None, last_act=None, **kwargs):
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path

        if self.visual_icl and not self.language_only:
            first_prompt, task_prompt = self.process_prompt_visual_icl(user_instruction, avg_obj_coord, prev_act_feedback=self.episode_act_feedback)
            if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL' in self.model_name or 'Qwen2.5-VL' in self.model_name:
                task_prompt += "\n\n"
                task_prompt = task_prompt + template_lang_manip if self.language_only else task_prompt + template_manip
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation, self.episode_messages)
                else:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)
            full_example_prompt = None
        else:
            full_example_prompt, task_prompt = self.process_prompt(
                self.system_prompt,
                user_instruction,
                avg_obj_coord,
                task_variation,
            )
            if (
                "claude" in self.model_name
                or "InternVL" in self.model_name
                or "Qwen2-VL" in self.model_name
                or "Qwen2.5-VL" in self.model_name
            ):
                task_prompt += "\n\n"
                task_prompt = task_prompt + template_lang_manip if self.language_only else task_prompt + template_manip
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt, last_act)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt, last_act, self.episode_messages)
                else:
                    self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt, last_act)

        actions, str_outs = self.query_vlm(obs, task_prompt, full_example_prompt, query_num, temperature)

        if self.chat_history:
            assert False, "Chat history is not supported in ManipPlanner yet."
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )

        self.planner_steps += 1
        # save the chat log
        self.save_chat_log(obs[0], str_outs)
        return actions, str_outs

    def save_chat_log(self, current_obs_path, response_texts, messages=None):
        if messages is None:
            messages = self.episode_messages
        chat_logs = []
        for response_text in response_texts:
            chat_log = format_chat_dialog(messages, response_text)
            chat_logs.append(chat_log)

        chat_path = os.path.join(os.path.dirname(current_obs_path), "chat_log.json")
        # append a timestamp to avoid overwriting
        timestamp = _dt.datetime.now().strftime("%m%d_%H%M%S")
        chat_path = chat_path.replace(".json", f"_{timestamp}.json")
        os.makedirs(os.path.dirname(chat_path), exist_ok=True)
        with open(chat_path, "w", encoding="utf-8") as f:
            json.dump(chat_logs, f, indent=2, ensure_ascii=False)

    def query_vlm(self, obs, task_prompt, full_example_prompt, query_num, temperature, parser_type="actions", **kwargs):
        """Query VLM and parser the output to correct format."""
        T = 0; max_try = 12
        collected_temp, parsered_jsons = [], []
        while T < max_try:
            try:
                outs = self.generate_model_response(
                    query_num=query_num,
                    temperature=temperature,
                )
                logger.info(f"==> Response from query_vlm:\n{outs}\n")
                assert isinstance(outs, list), "Model response should be a list of outputs."

                for out in outs:
                    try:
                        if parser_type == "actions":
                            actions, json = self._parser_action_plan(out, self.action_length)
                        elif parser_type == "reward":
                            actions, json = self._parser_evaluation(out, kwargs["candidate_idxs"])

                    except Exception as e:
                        logger.info("-" * 100)
                        traceback.print_exc()
                        logger.info("-" * 100)
                        continue

                    collected_temp.append(actions)
                    parsered_jsons.append(json)
                    if len(collected_temp) >= query_num:
                        break

                if len(collected_temp) < query_num:
                    logger.info("Not enough instructions, retrying...")
                    raise ValueError("Not enough instructions")

                return collected_temp, parsered_jsons
            except Exception as e:
                logger.info("-" * 100)
                traceback.print_exc()
                logger.info("-" * 100)
                self._bump_error()
                T += 1
                time.sleep(1 * T)  # Exponential backoff

        # Finally, if we still fail to get a valid action, return a random action
        logger.error(f"Failed to get a valid action after {T} attempts, returning a random action.")
        return [[self._random_action(),],], None

    def _parser_action_plan(self, out, action_length):
        actions, json_output = self.json_to_action(out)
        # Ensure actions is a list of lists of ints with shape [n, 7]
        actions = np.array(actions, dtype=int).reshape(-1, action_length).tolist()
        logger.info("Collected an effective action plan")
        return actions, json_output

    def _parser_evaluation(self, out, candidate_idxs):
        # * 1. try to parse the output text as JSON
        json_object = repair_json(out, return_objects=True)
        keys = ("reasoning", "current_best_plan", "fully_achieved")
        choice_idx = json_object["current_best_plan"]
        assert all(key in json_object for key in keys), "The output JSON is missing some items"
        assert isinstance(choice_idx, int) and choice_idx in candidate_idxs, "The current_best_plan should be an integer in the candidate_idxs"
        assert isinstance(json_object["fully_achieved"], bool), "The fully_achieved should be a boolean value"

        logger.info("Collected an effective evaluation")
        return choice_idx, json_object

    def act_custom(self, prompt, obs):
        assert type(obs) == str  # input image path
        out = self.model.respond(prompt, obs)
        out = out.replace("'", '"')
        out = out.replace('\"s ', "\'s ")
        out = out.replace('```json', '').replace('```', '')
        return out

    def generate_model_response(self, query_num=1, **kwargs):
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        out = self.model.respond(self.episode_messages, n=query_num, **kwargs)
        return out

    def update_info(self, info):
        super().update_info(info)
        env_feedback = info['env_feedback']
        action = info['action']
        self.episode_act_feedback.append([action, env_feedback])
