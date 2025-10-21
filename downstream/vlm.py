import asyncio
import base64
import json
import math
import os.path as osp
from typing import Literal
import ast
import traceback

import numpy as np
import tiktoken
from openai import AsyncOpenAI, OpenAI
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from PIL import Image
from pydantic import BaseModel
import re

# response = ParsedChatCompletion.model_validate(response)
from tqdm.asyncio import tqdm_asyncio
from utils.util import is_empty
from downstream.utils.util import output_is_none
from downstream.utils.query_utils import extract_lengths, process_error
from typing import Any, Dict, List, Sequence, Union
from json_repair import repair_json


WORLD_MODEL_TYPES = {
    "text": ["wan21", "ltx", "hunyuan", "nwm", "cosmos", "wan22", "svd", "gen4tur"],    # zero-shot models
    "FTtext": ["FTcosmos", "FTltx", "FTwan21", "FTwan22", "FTwan22-14B"],    # post-trained (finetuned) models
    "camera": ["se3ds", "pathdreamer"],    # zero-shot models
    "action": ["igen"],    # post-trained models
    "GTsim": ["GTsim"],
}

VIEW_ORDER = ("front", "left", "right", "back")
VIEW_ID_OFFSET = {0: 0.0, 1: -0.5 * np.pi, 2: 0.5 * np.pi, 3: np.pi}


FIELD_NAME = "c"  # NOTE: for saving output tokens
# MSG_KEYS = ["role", "message"]
# HISTORY_FILE = "history.jsonl"
LOCAL_MODELS = [
    "llama3.2",
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    "OpenGVLab/InternVL3-78B-AWQ",
]
COMMERCIAL_MODELS = [
    "gpt-4o-mini-2024-07-18",
    "gemini-2.0-flash-lite-preview-02-05",
    "gpt-4o"
]


CHOICES = dict(
    digit=[f"{i:03d}" for i in range(1000)],  # length: 1000, best for OpenAI models
    letter=(
        "".join(chr(i) for i in range(ord("A"), ord("Z") + 1))
        + "ΓΔΘΛΞΠΣΦΨΩ"  # length: 36, best for smaller models
    ),
)
CHOICE_FORMATS = dict(
    digit="the 3-digit number",
    letter="the choice of a single capital letter",
)

EXTRA_CHAT_ARGS = {"top_k": -1, "top_p": 1.0, "temperature": 1.0}       # tell vLLM “no hard cap”


def create_category_set(
    categories: list[str],
    choice_format: Literal["digit", "letter"],
) -> type[BaseModel]:
    """
    - Input: ["apple", "banana", "cherry"]
    - Output: category_set = create_category_set(["apple", "banana", "cherry"])

    To recover choices from the instance: category_set.__annotations__[FIELD_NAME].__args__
    """
    assert (
        len(categories) <= 1000
    ), "OpenAI vocabulary only supports 000-999 as a single token, 4-digit text will be splitted."

    choices = [
        f"{choice}: {cat}" for choice, cat in zip(CHOICES[choice_format], categories)
    ]
    # For Python 3.11
    # literal_type = Literal[*choices]
    literal_type = Literal.__getitem__(tuple(choices))
    return type(
        "CategorySet",
        (BaseModel,),
        {"__annotations__": {FIELD_NAME: literal_type}},
    )


class LMTokenizer:
    def __init__(self, model):
        self.tokenizer = tiktoken.encoding_for_model(model)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def tokenize(self, text):
        token_ids = self.encode(text)
        tokens = [self.decode([token_id]) for token_id in token_ids]
        return tokens


class LMClassifier:

    def __init__(
        self, api_key, model, categories: list[str], top_logprobs, base_url=None
    ):
        self.api_key = api_key
        if base_url is None:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        # self.aclient = AsyncOpenAI(api_key=api_key)
        self.model = model
        # self.client.base_url = 'https://api.onechats.top/v1'
        self.temperature = 1.0  # 0.1, 1
        self.image_detail = "high"  # "low", "high"
        self.top_logprobs = top_logprobs  # NOTE: max is 20
        assert self.top_logprobs <= 20
        if model in COMMERCIAL_MODELS:
            self.choice_format = "digit"
            self.tokenizer = LMTokenizer(model)
            self.clf_index = len(self.tokenizer.encode('{"' + FIELD_NAME + '":"'))
        elif model in LOCAL_MODELS:
            self.choice_format = "letter"
            assert len(categories) <= len(CHOICES[self.choice_format])
            self.clf_index = 0
        else:
            raise ValueError(f"Invalid model: {model}")
        self.categories = categories
        if len(self.categories) != 0:
            self.response_format = create_category_set(categories, self.choice_format)

    # ------------------------------------------------------------------ #
    # Specific Query methods acording to task
    # ------------------------------------------------------------------ #
    # ---- query methods section (0) ---- #
    def __query_classify(self, messages):
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            response_format=self.response_format,
        )
        print(response.choices[0].message.content)
        return response.choices[0].logprobs.content

    def __query_classify_plain(self, messages):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                    extra_body=EXTRA_CHAT_ARGS | {"temperature": self.temperature},
                )
                print(f"Response text: {response.choices[0].message.content}")
                return response.choices[0].logprobs.content
            except Exception as e:
                print("-" * 100)
                traceback.print_exc()

    def __query_text(self, messages, n=1):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            n=n,
            extra_body=EXTRA_CHAT_ARGS | {"temperature": self.temperature},
        )
        response = [response.choices[i].message.content for i in range(n)]
        for res in response:
            print(f"Response text: {res}")
        return response

    def __query_text_perplexity(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
        )
        print(f"Response text: {response.choices[0].message.content}")
        answer = response.choices[0].message.content
        # * compute negative perplexity (average token probability)
        logprobs = response.choices[0].logprobs.content
        logprobs = [token.logprob for token in logprobs]
        avg_logprob = sum(logprobs) / len(logprobs)
        answer_value = math.exp(avg_logprob)
        return {answer: answer_value}

    def __get_classify_logprobs(self, logprobs, return_prob, trunc_thres=None):
        valid_choices = CHOICES[self.choice_format][:len(self.categories)]
        clf_logprobs = logprobs[self.clf_index].top_logprobs
        # clf_logprobs = clf_logprobs[: len(self.categories)]
        clf_logprobs = [x for x in clf_logprobs if x.token in valid_choices]
        if trunc_thres is not None:
            clf_logprobs = [x for x in clf_logprobs if math.exp(x.logprob) >= trunc_thres]
        assert len(clf_logprobs) != 0, "No valid choices"
        clf_logprobs = {
            self.categories[valid_choices.index(x.token)]: x.logprob
            for x in clf_logprobs
        }
        if return_prob:
            for token, logprob in clf_logprobs.items():
                clf_logprobs[token] = math.exp(logprob)

        # sort by logprob in descending order
        clf_logprobs_sort = sorted(
            clf_logprobs.items(), key=lambda x: x[1], reverse=True
        )
        # convert to dict
        clf_logprobs = {k: v for k, v in clf_logprobs_sort}

        return clf_logprobs

    # ---- query methods section (1) ---- #
    def _classify(self, messages, return_prob, **kwargs):
        logprobs = self.__query_classify(messages, **kwargs)
        logprobs = self.__get_classify_logprobs(logprobs, return_prob)
        # self.save_chat()
        return logprobs

    def _classify_plain(self, messages, return_prob, **kwargs):
        """
        NOTE: open source models may not strictly follow the response format.
        """
        T = 5
        while T > 0:
            try:
                logprobs = self.__query_classify_plain(messages, **kwargs)
                logprobs = self.__get_classify_logprobs(logprobs, return_prob)
                return logprobs
            except Exception as e:
                print("-" * 100)
                traceback.print_exc()
                # print(messages[0]['content'][0]['text'])
                xs = logprobs[self.clf_index].top_logprobs
                answers = [f"P({x.token})={math.exp(x.logprob):5.1%}" for x in xs]
                valid_answers = [x.token in CHOICES[self.choice_format] for x in xs]
                print(f"Valid choices: {CHOICES[self.choice_format]}")
                # randome select a choice, if T=0:
                print(f"Answer logprobs: {answers}")
                print(f"Valid answer: {valid_answers}")
                T -= 1
        # Finally, if we still fail to get a valid output
        print(f"WARNING: Reach max try, Return Random choice: {np.random.choice(self.categories)}")
        return dict(zip(self.categories, [1/len(self.categories)] * len(self.categories)))

    # ---- Parser methods ---- #
    def __parser_action_seq(self, text_seqs, split_char=","):
        # action_choices = {i: [] for i in range(self.look_ahead_action_num)}
        action_seqs = []
        for i, raw_answer in enumerate(text_seqs):
            if is_empty(raw_answer):
                continue
            raw_answer = raw_answer.replace("\\","")
            answer = raw_answer[raw_answer.rfind("["):raw_answer.rfind("]")+1]
            answer = ast.literal_eval(answer)
            assert 1 <= len(answer) <= self.look_ahead_action_num, f"Answer length exceeds max length {self.look_ahead_action_num}"

            is_stop = False
            if len(answer) == 1 or len(set(answer)) == 1:
                if "stop" == self.categories[CHOICES[self.choice_format].index(answer[0])]:
                    is_stop = True
            else:
                if "stop" == self.categories[CHOICES[self.choice_format].index(answer[-1])]:
                    #remove the last element if it is 'stop'
                    answer = answer[:-1]

            action_seq_str = []
            for letter in answer:
                action_seq_str.append(
                    self.categories[CHOICES[self.choice_format].index(letter)]
                )
            action_seqs.append({
                "origin_answer": answer,
                "convert_answer": action_seq_str,
                "is_stop": is_stop,
                "seq_len": len(answer)
            })

        return action_seqs

    def __parser_lowlevel_actions(self, text_seqs):
        necessary_keys = {'Chosen Direction Mark', 'Forward Number'}
        action_seqs = []
        for i, raw_answer in enumerate(text_seqs):
            if is_empty(raw_answer):
                continue
            raw_answer = raw_answer.replace("\\","")
            answer = raw_answer[raw_answer.rfind("{"):raw_answer.rfind("}")+1]
            answer = ast.literal_eval(answer)
            if set(answer.keys()) != necessary_keys:
                raise ValueError("Invalid answer format")

            assert (
                0 <= answer["Forward Number"] <= self.look_ahead_action_num
            ), f"Answer length exceeds max length {self.look_ahead_action_num}"
            assert (answer["Chosen Direction Mark"] in self.categories or \
                output_is_none(answer["Chosen Direction Mark"])
            ), f"Invalid direction mark: {answer['Chosen Direction Mark']}"

            action_seqs.append(answer)
        return action_seqs

    def __parser_highlevel_plan(self, text_seqs, input, split_char="{}"):
        # action_choices = {i: [] for i in range(self.look_ahead_action_num)}
        answer_key = self.get_answer_key_for_return()
        necessary_keys = {
            "Reason",
            "Action Plan",
            "Chosen View",
            "Chosen Landmark",
            answer_key,
        }

        instructions = []
        given_obj_ids = input['detected_objs']
        all_obj_ids = [list(v.keys()) for view_name, v in given_obj_ids.items()]
        for i, raw_answer in enumerate(text_seqs):
            if is_empty(raw_answer):
                continue
            # raw_answer = raw_answer.replace("\\","")
            # answer = raw_answer[raw_answer.rfind("{"):raw_answer.rfind("}")+1]
            # answer = ast.literal_eval(answer)
            answer = repair_json(raw_answer, return_objects=True)

            if set(answer.keys()) != necessary_keys:
                raise ValueError("Invalid answer format")

            chosen_id, chosen_view, landmark_ori_idxs = self.__parser_id_and_view(
                all_obj_ids, answer
            )
            if output_is_none(answer[answer_key]):
                query_answer = None
            else:
                query_answer = answer[answer_key]

            answer[answer_key] = query_answer
            answer['Chosen Landmark'] = chosen_id
            answer['Chosen View'] = chosen_view
            answer['landmark_ori_idxs'] = landmark_ori_idxs

            # match the words like 'left view', 'right view', 'front view', 'back view', and replace them with "current view"
            action_plan_text = answer['Action Plan']
            pattern = r'(^|\s)(["\']?)(left|right|front|back)(["\']?)\s+view\b'
            replacement = r"\1current view"
            new_action_plan = re.sub(pattern, replacement, action_plan_text, flags=re.IGNORECASE)
            answer['Action Plan'] = new_action_plan
            instructions.append(answer)

        return instructions

    def __parser_id_and_view(self, all_obj_ids, answer):
        # parser 'Chosen Landmark' as chosen_id:
        landmark_ori_idxs, landmark_view_idxs = [], []
        if not output_is_none(answer['Chosen Landmark']):
            if isinstance(answer['Chosen Landmark'], str):
                chosen_id = int(answer['Chosen Landmark'])
            else:
                chosen_id = answer['Chosen Landmark']
            for view_idx, view_objs in enumerate(all_obj_ids):
                for obj_idx, curr_id in enumerate(view_objs):
                    if curr_id == chosen_id:
                        landmark_ori_idxs.append((view_idx, obj_idx))
                        landmark_view_idxs.append(view_idx)
        else:
            chosen_id = None

        # parser 'Chosen View':
        chosen_view = answer['Chosen View']
        assert chosen_view in VIEW_ORDER, f"Invalid view: {chosen_view}"
        chosen_view_idx = VIEW_ORDER.index(chosen_view)
        if not is_empty(chosen_id):
            # assert len(landmark_view_idxs) > 0
            assert chosen_view_idx in landmark_view_idxs, \
                f"Inconsistent view vs landmark: {chosen_view_idx} vs {landmark_view_idxs}. " \
                f"Current chosen_view is {chosen_view}, and chosen_id is {chosen_id}."
        return chosen_id, chosen_view, landmark_ori_idxs

    # ---- query methods section (2) ---- #
    def _query_next_N_action(self, messages, N, **kwargs):
        """
        NOTE: open source models may not strictly follow the response format.
        """
        T = 30; collected_temp = []
        while T > 0:
            try:
                text = self.__query_text(messages, n=N, **kwargs)
                action_seqs = self.__parser_action_seq(text)

                for action_seq in action_seqs:
                    if self._is_valide_actions_for_task(action_seq["seq_len"]):
                        collected_temp.append(action_seq)
                        print("Collected an effective action sequence")
                    if len(collected_temp) >= N:
                        break

                if len(collected_temp) < N:
                    print("Not enough instructions, retrying...")
                    raise ValueError("Not enough instructions")

                action_seqs = collected_temp
                if is_empty(action_seqs):
                    raise ValueError("Empty action sequences")
                return action_seqs
            except Exception as e:
                print("-" * 100)
                traceback.print_exc()

                error_message = str(e)
                current_length, context_length = extract_lengths(error_message)

                if current_length and context_length:
                    num_message = len(messages)

                    # Estimate how many messages to remove
                    ratio = context_length / current_length
                    suitable_num = int(num_message * ratio)
                    num_to_remove = num_message - suitable_num
                    # make sure the num_to_remove is a even num
                    num_to_remove = num_to_remove + 1 if num_to_remove % 2 == 1 else num_to_remove
                    num_to_remove = max(2, num_to_remove)

                    if num_message > num_to_remove + 1:
                        print(f"Prompt too long, removing {num_to_remove} earliest messages and retrying...")
                        messages_preserve = messages[:3]
                        messages_ = messages[3+num_to_remove:]  # Remove earliest messages
                        messages = messages_preserve + messages_
                    else:
                        print("WARNING: Cannot reduce messages further, returning empty response")
                        break
                else:
                    messages = process_error(error_message, messages, T)
                T -= 1
        # Finally, if we still fail to get a valid output
        print("WARNING: Reach max retry limit, returning empty response")
        return []

    def _query_next_instruction(self, messages, N, input):
        """
        NOTE: open source models may not strictly follow the response format.
        """
        T = 30; collected_temp = []
        while T > 0:
            try:
                texts = self.__query_text(messages, n=N)
                for text in texts:
                    try:
                        instruction = self.__parser_highlevel_plan([text], input)
                    except Exception as e:
                        print("-" * 100)
                        traceback.print_exc()
                        continue
                    collected_temp.extend(instruction)
                    print("Collected an effective instruction")
                    if len(collected_temp) >= N:
                        break

                if len(collected_temp) < N:
                    print("Not enough instructions, retrying...")
                    raise ValueError("Not enough instructions")

                instructions = collected_temp
                if is_empty(instructions):
                    raise ValueError("Empty action sequences")
                return instructions

            except Exception as e:
                print("-" * 100)
                traceback.print_exc()

                error_message = str(e)
                current_length, context_length = extract_lengths(error_message)

                if current_length and context_length:
                    num_message = len(messages)

                    # Estimate how many messages to remove
                    ratio = context_length / current_length
                    suitable_num = int(num_message * ratio)
                    num_to_remove = num_message - suitable_num
                    # make sure the num_to_remove is a even num
                    num_to_remove = num_to_remove + 1 if num_to_remove % 2 == 1 else num_to_remove
                    num_to_remove = max(2, num_to_remove)

                    if num_message > num_to_remove + 1:
                        print(f"Prompt too long, removing {num_to_remove} earliest messages and retrying...")
                        messages = messages[num_to_remove:]  # Remove earliest messages
                    else:
                        print("WARNING: Cannot reduce messages further, returning empty response")
                        break
                else:
                    messages = process_error(error_message, messages, T)
                T -= 1
        # Finally, if we still fail to get a valid output
        print("WARNING: Reach max retry limit, returning empty response")
        instructions = collected_temp
        return instructions

