import os
import re
import base64
import copy
from mimetypes import guess_type
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
import typing_extensions as typing
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
import io
from typing import List, Sequence, Union, Any, Dict, Set
Pose = Sequence[float]           # length-7 (x, y, z, qx, qy, qz, qw)
PoseBatch = Sequence[Pose]       # list/tuple/array of poses

template_lang = '''\
The output json format should be {'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, \
2. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, \
3. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!
'''

template = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image,
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task,
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name,
4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

template_lang_manip = '''\
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':str}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection: Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available.
2. language_plan: A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name (refer to objects by name, not index in your language plan).
3. executable_plan: A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

template_manip = '''\
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':str}
The fields in above JSON follows the purpose below:
1. visual_state_description: Describe the color and shape of each object according to  the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input.
2. reasoning_and_reflection: Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available.
3. language_plan: A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name (refer to objects by name, not index in your language plan).
4. executable_plan: A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.
5. keep your plan efficient and concise.
'''


def _plan_to_key(plan):
    """
    Convert a plan (List[np.ndarray] or List[List[float]]) into an
    immutable, hashable tuple so it can be counted or used as a
    dictionary key.

    Each action a_t ∈ ℝ⁷ is mapped to tuple(a_t).  The full plan
    becomes a tuple of those tuples.
    """
    return tuple(tuple(np.asarray(step, dtype=float)) for step in plan)


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)

# --------------------------------------------------------------------------------------
# Quaternion helpers
# --------------------------------------------------------------------------------------
def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Return *q* scaled to unit length (handles zeros gracefully)."""
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("Quaternion has near‑zero length; can’t normalise.")
    return q / norm


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions *q0* and *q1* at *t*∈[0,1]."""
    # Ensure shortest path
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Clamp dot to stay in domain of arccos/√
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:  # quats are almost identical – use lerp + renorm for numerical stability
        q = q0 + t * (q1 - q0)
        return _normalize_quat(q)

    theta_0 = np.arccos(dot)         # angle between q0 and q1
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1

# --------------------------------------------------------------------------------------
# Pose interpolation (xyz + quat)
# --------------------------------------------------------------------------------------
def interpolate_7dof_pose(
    start_pose: Union[Pose, PoseBatch],
    end_pose: Union[Pose, PoseBatch],
    num_points: int,
    include_end: bool = True,
) -> List[List[float]]:
    """
    Interpolate between one or many 7-DoF poses.

    Parameters
    ----------
    start_pose, end_pose
        • Batch  mode  : iterable of N length-7 sequences.
    num_points
        Number of points *per pair* (≥2).
    include_end
        If False, the last pose (t=1.0) of every segment is dropped.

    Returns
    -------
    out
        list of poses = [batch_size, (num_points or num_points-1)].
    """
    # ---------- 1. Normalise input to *lists of numpy arrays* ---------------------------
    def as_np(p: Union[Pose, PoseBatch]) -> List[np.ndarray]:
        return [np.asarray(x, dtype=float) for x in p]

    s_batch = as_np(start_pose)
    e_batch = as_np(end_pose)

    if len(s_batch) != len(e_batch):
        raise ValueError(f"start_pose batch size {len(s_batch)} != end_pose batch size {len(e_batch)}")
    if num_points < 2:
        raise ValueError("num_points must be ≥ 2")

    batch_size = len(s_batch)
    t_vals = np.linspace(0.0, 1.0, num_points)
    if not include_end:
        t_vals = t_vals[:-1]           # drop 1.0

    # ---------- 2. Interpolate each pair ------------------------------------------------
    outs: List[List[float]] = []
    for p0, p1 in zip(s_batch, e_batch):
        xyz0, xyz1 = p0[:3], p1[:3]
        q0,  q1   = _normalize_quat(p0[3:]), _normalize_quat(p1[3:])

        out: List[List[float]] = []
        for t in t_vals:
            xyz  = (1.0 - t) * xyz0 + t * xyz1
            quat = _slerp(q0, q1, t)
            out.append([*xyz, *quat])
        outs.append(out)

    return np.asarray(outs, dtype=float)

#-------------------------- End New ------------------------------

def fix_json(json_str):
    """
    Locates the substring between the keys "reasoning_and_reflection" and "language_plan"
    and escapes any inner double quotes that are not already escaped.

    The regex uses a positive lookahead to stop matching when reaching the delimiter for the next key.
    """
    # first fix common errors
    json_str = json_str.replace("'",'"')
    json_str = json_str.replace('\"s ', "\'s ")
    json_str = json_str.replace('\"re ', "\'re ")
    json_str = json_str.replace('\"ll ', "\'ll ")
    json_str = json_str.replace('\"t ', "\'t ")
    json_str = json_str.replace('\"d ', "\'d ")
    json_str = json_str.replace('\"m ', "\'m ")
    json_str = json_str.replace('\"ve ', "\'ve ")
    json_str = json_str.replace('```json', '').replace('```', '')

    # Then fix some situations. Pattern explanation:
    # 1. ("reasoning_and_reflection"\s*:\s*") matches the key and the opening quote.
    # 2. (?P<value>.*?) lazily captures everything in a group named 'value'.
    # 3. (?=",\s*"language_plan") is a positive lookahead that stops matching before the closing quote
    #    that comes before the "language_plan" key.
    pattern = r'("reasoning_and_reflection"\s*:\s*")(?P<value>.*?)(?=",\s*"language_plan")'

    def replacer(match):
        prefix = match.group(1)            # Contains the key and the opening quote.
        value = match.group("value")         # The raw value that might contain unescaped quotes.
        # Escape any double quote that is not already escaped.
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + fixed_value

    # Use re.DOTALL so that newlines in the value are included.
    fixed_json = re.sub(pattern, replacer, json_str, flags=re.DOTALL)
    return fixed_json


class ExecutableAction_1(typing.TypedDict):
    action_id: int = Field(
        description="The action ID to select from the available actions given by the prompt"
    )
    action_name: str = Field(
        description="The name of the action"
    )
class ActionPlan_1(BaseModel):
    visual_state_description: str = Field(
        description="Description of current state from the visual image"
    )
    reasoning_and_reflection: str = Field(
        description="summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task"
    )
    language_plan: str = Field(
        description="The list of actions to achieve the user instruction. Each action is started by the step number and the action name"
    )
    executable_plan: list[ExecutableAction_1] = Field(
        description="A list of actions needed to achieve the user instruction, with each action having an action ID and a name."
    )

class ActionPlan_1_manip(BaseModel):
    visual_state_description: str = Field(
        description="Describe the color and shape of each object according to  the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input."
    )
    reasoning_and_reflection: str = Field(
        description="Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available."
    )
    language_plan: str = Field(
        description="A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name."
    )
    executable_plan: str = Field(
        description="A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action."
    )

def convert_format_2claude(messages):
    new_messages = []

    for message in messages:
        if message["role"] == "user":
            new_content = []

            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)

    return new_messages

def convert_format_2gemini(messages):
    new_messages = []

    for message in messages:
        if message["role"] == "user":

            new_content = []
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_data}"
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)

    return new_messages



class ExecutableAction(typing.TypedDict):
    action_id: int
    action_name: str
class ActionPlan(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction]

class ActionPlan_manip(BaseModel):
    visual_state_description: str
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: str

class ExecutableAction_lang(typing.TypedDict):
    action_id: int
    action_name: str
class ActionPlan_lang(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: list[ExecutableAction_lang]

class ActionPlan_lang_manip(BaseModel):
    reasoning_and_reflection: str
    language_plan: str
    executable_plan: str

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def chw_list_to_data_url(image_chw, mime_type="image/png"):
    # list → numpy
    arr = np.array(image_chw, dtype=np.float32)

    if arr.max() <= 1.0:
        arr = (arr * 255).clip(0, 255)
    arr = arr.astype(np.uint8)

    arr = np.transpose(arr, (1, 2, 0))

    image = Image.fromarray(arr)

    buffer = io.BytesIO()
    image.save(buffer, format=mime_type.split("/")[-1].upper())

    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"