vlm_generation_guide_manip={
    "type": "object",
    'properties': {
        "visual_state_description": {
            "type": "string",
            "description": "Describe the color and shape of each object in the detection box in the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input.",
        },
        "reasoning_and_reflection": {
            "type": "string",
            "description": "Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available.",
        },
        "language_plan": {
            "type": "string",
            "description": "A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name.",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The 7-dimensional discrete action in the format of a list given by the prompt",
                    }
                },
                "required": ["action"]
            }
        },
    },
    "required": ["visual_state_description", "reasoning_and_reflection", "language_plan", "executable_plan"]
}

llm_generation_guide_manip={
    "type": "object",
    'properties': {
        "reasoning_and_reflection": {
            "type": "string",
            "description": "Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available.",
        },
        "language_plan": {
            "type": "string",
            "description": "The list of actions to achieve the user instruction. Each action is started by the step number and the action name.",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of actions needed to achieve the user instruction, with each action being a 7-dimensional discrete action in the format of a list.",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The 7-dimensional discrete action in the format of a list given by the prompt",
                    }
                },
                "required": ["action"]
            }
        },
    },
    "required": ["reasoning_and_reflection", "language_plan", "executable_plan"]
}

Igenex_generation_guide_manip={
    "type": "object",
    'properties': {
        "description": {
            "type": "string",
            "description": "Describe the color, shape and potential usage of each object in the numerical order in the image.",
        },
        "language_plan": {
            "type": "string",
            "description": "Reason about the overall plan that needs to be taken on the target objects. Give a list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name.",
        },
        "reasoning_and_reflection": {
            "type": "string",
            "description": "Reflect on which provided prediction of future are most likely to complete the task, and why.",
        },
        "action_sequence_choice": {
            "type": "integer",
            "description": "One integer, representing the index of the action sequence to choose from the provided predictions.",
        },
    },
    "required": ["visual_state_description", "language_plan", "reasoning_and_reflection", "action_sequence_choice"]
}



Igenex_evaluator_guide={
    "type": "object",
    'properties': {
        "language_plan": {
            "type": "string",
            "description": "Reason about the overall plan that needs to be taken on the target objects. Give a list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name.",
        },
        "reasoning": {
            "type": "string",
            "description": "Reflect on which provided prediction of future are most likely to complete the task, and why.",
        },
        "current_best_plan": {
            "type": "integer",
            "description": "One integer, representing the index of the action sequence to choose from the provided predictions.",
        },
        "fully_achieved": {
            "type": "boolean",
            "description": "Whether the current best plan fully achieves the task goal.",
        },
        "why_not_fully_achieved": {
            "type": "string",
            "description": "Reason why the current best plan does not fully achieve the goal, if applicable.",
        },
        "how_to_improve": {
            "type": "string",
            "description": "Suggestions, analysis, and notes for improving the current best plan if it does not fully achieve the goal.",
        },
    },
    "required": ["language_plan", "reasoning", "current_best_plan", "fully_achieved", "why_not_fully_achieved", "how_to_improve"]
}


Igenex_revise_auxiliary_prompt = '''
You are now provided with **simulated outcomes** in addition to your real-time observations. These outcomes are low-resolution predictions of what the scene may look like after executing hypothetical action plans. You should choose the best plan based on these predictions and return the index of it.

They are intended to help you reason about the environment and make more informed decisions.

### Simulated Outcome Structure
Each simulated-outcome item includes a sequence of pictures of the simulated results after following the proposed plan.

### How to Use This Information
You must consider both:
1. Your current **real observation** of the environment, and
2. The provided **simulated outcomes**.

Use these to:
- Evaluate how well each proposed plan satisfies the task objective.
- Identify if any proposed plan fully achieves the instruction goal.
- If a proposed plan appears valid and effective, **you may adopt it directly** as your final response.

### Additional Notes
- Simulated outcomes are **approximate**. Treat them as helpful forecasts, not absolute truth.
- You must analyze these hypothetical action plans and their simulated outcomes in the `reasoning` field of the returned JSON (e.g., their differences and why you choose one over another).
- Always prioritize correctness and robustness in the final executable plan.
'''


Igenex_evaluator_prompt = '''
You are a trajectory evaluator for a robotic manipulation task. Your job is to assess candidate trajectories and select the most suitable one, if any, for real-world execution.

## Inputs
For each evaluation, you are given:
1. **Task Instructions**: A natural language description of the desired manipulation outcome.
2. **Candidate Action Plans (Described Visually)**: Each trajectory is detailed in natural language, including:
   - Object properties (e.g., color, shape)
   - Spatial relationships (e.g., positions, orientations)
   - Contact points and interactions
   - Gripper state and final pose

## Your Evaluation Task
You must:
- Analyze all candidate trajectories carefully.
- Decide whether any trajectory **achieves** or **clearly progresses toward** the task goal.
  - If one or more satisfy the goal, choose the **best** (prioritize correctness and efficiency).
  - If it is clear that none of the trajectories achieve or progress toward the goal, set `current_best_plan` to -1.

Provide justification by:
- Citing concrete visual descriptions (e.g., object positions, orientations, contact points, gripper state and pose).
- Explaining how the chosen plan aligns with the task goal or, if incomplete, why it is still the best available option.

## Output Format
Return **only** the following JSON object (no extra text, no markdown):
```
{"task_goal": "<One-sentence restatement of the manipulation objective in clear natural language.>",
"reasoning": "<Break the goal into substeps, then compare candidates step by step using specific visual evidence. Explain which steps succeed or fail and why. Conclude with a short rationale for the selected plan.>",
"current_best_plan": <integer index of the chosen trajectory (0-based), or -1 if none are suitable>
}
```

## Additional Notes
- If few-shot examples are provided, follow their analysis process and use them as a reference to calibrate how to distinguish strong from weak action plans.
- Remain objective. Base your analysis strictly on the provided visuals and task instructions.
'''

Igenex_evaluator_prompt_norepeat = '''
You are a trajectory evaluator for a robotic manipulation task. Your job is to assess candidate trajectories and select the most suitable one, if any, for real-world execution.

## Inputs
For each evaluation, you are given:
1. **Task Instructions**: A natural language description of the desired manipulation outcome.
2. **Candidate Action Plans (Described Visually)**: Each trajectory is detailed in natural language, including:
   - Object properties (e.g., color, shape)
   - Spatial relationships (e.g., positions, orientations)
   - Contact points and interactions
   - Gripper state and final pose

## Your Evaluation Task
You must:
- Analyze all candidate trajectories carefully.
- Decide whether any trajectory **achieves** or **clearly progresses toward** the task goal.
  - If one or more satisfy the goal, choose the **best** (prioritize correctness and efficiency).
  - If none fully satisfy the goal, choose the trajectory that gets **closest** to success.

Provide justification by:
- Citing concrete visual descriptions (e.g., object positions, orientations, contact points, gripper state and pose).
- Explaining how the chosen plan aligns with the task goal or, if incomplete, why it is still the best available option.

## Output Format
Return **only** the following JSON object (no extra text, no markdown):
```
{"task_goal": "<One-sentence restatement of the manipulation objective in clear natural language.>",
"reasoning": "<Break the goal into substeps, then compare candidates step by step using specific visual evidence. Explain which steps succeed or fail and why. Conclude with a short rationale for the selected plan.>",
"current_best_plan": <integer index of the chosen trajectory (0-based)>
}
```

## Additional Notes
- If few-shot examples are provided, follow their analysis process and use them as a reference to calibrate how to distinguish strong from weak action plans.
- Remain objective. Base your analysis strictly on the provided visual descriptions and task instructions.
'''

Igenex_descriptor_prompt = '''
You are a trajectory descriptor for a robotic manipulation task. Your role is to examine several candidate trajectories generated in simulation and provide clear natural-language descriptions of the scene and the action trajectories.

## Inputs
For each evaluation you receive:
- **Candidate Trajectories**: A sequence of key-frame images showing the action plan over time, where each frame represents the predicted scene after one action step.

## Your Task
You must:
- Carefully analyze all candidate trajectories.
- Describe the scene objects and their positions in natural language, including details such as object colors, shapes, relative positions, orientations, contact points, and the gripper’s state and pose.

## Output Format
Return **only** the following JSON object (no extra text, no markdown):
```
{"scene_description": "<Natural-language description of the objects: list their colors, shapes, and positions in the scene.>",
"action_trajectory_description": "<Objective description of each candidate trajectory: describe object interactions, relative positions, contact events, and gripper pose/state at key moments.>"
}
```

## Additional Notes
- If few-shot examples are provided, follow their style closely and use them as references for the required level of detail, phrasing, and format (e.g., how to describe object positions and the end-effector state). Match the examples in style and completeness.
- Remain objective. Base your descriptions strictly on the provided visuals.
'''


# ========================= Few-shot examples for VLM evaluator =========================
# =========================
# Push buttons
# =========================
E_push_buttons_task_goal = {"E1": "Press the button with the blue base."}

E_push_buttons_scene_description = {
    "E1": (
        "Initial scene: a tabletop with three square bases, each supporting a red circular button cap—blue-base at the center, yellow-base at front-right, and green-base at front-left."
    )
}

E_push_buttons_traj_description = {
    "E1": [
        "Candidate Action Plan <0>: the end-effector approaches diagonally toward the blue-base button and is currently closest to it.",
        "Candidate Action Plan <1>: the end-effector descends vertically toward the green-base button and is closest to it.",
        "Candidate Action Plan <2>: the end-effector translates rightward toward the yellow-base button and is closest to it.",
    ]
}

E_push_buttons_reasoning = {
    "E1": (
        "The task goal is to press the button with the blue base (note that all three buttons share the same red circular cap, so the base color identifies the target). "
        "To succeed, the end-effector must move downward to contact and press the correct button. "
        "Action Plan <0> approaches the blue-base button and is currently closest to it, which indicates it will produce the required pressing motion. "
        "Action Plan <1> moves toward and is now closest to the green-base button, which is not the target, so it does not meet the goal. "
        "Action Plan <2> moves toward and is now closest to the yellow-base button, which is also not the target, so it does not meet the goal. "
        "Among the three trajectories, Action Plan <0> is both closest to the target button and consistent with the required pressing motion. "
        "Therefore, Action Plan <0> is the best choice."
    )
}

E_push_buttons_best_plan = {"E1": 0}


# =========================
# Slide block → color target
# =========================
E_slide_block_to_color_target_task_goal = {
    "E1": "Slide the red block onto the magenta (purple) target square.",
}

_E_slide_block_to_color_target_scene_description = {
    "E1": (
        "Initial scene: a tabletop with colored markers and blocks. A green square sits at left-middle. "
        "A red block (cube) is near center-left, a blue square is upper-left, a magenta square is mid-right, and a yellow square is lower-right. The robot wrist is above the center column with the gripper open and pitched downward."
    )
}

_E_slide_block_to_color_target_traj_description = {
    "E1": [
        "Candidate Action Plan <0>: the end-effector approaches from the top-right and makes side contact with the red block on its right face (the red block ends up between the end-effector and the green square).",
        "Candidate Action Plan <1>: the end-effector descends near the red block at center-left and makes side contact with the red block on its left face (the red block ends up between the end-effector and the magenta square).",
        "Candidate Action Plan <2>: the end-effector lowers to approach the red block from above and makes contact on the back face of the red block (the red block ends up between the end-effector and the yellow square).",
    ]
}

_E_slide_block_to_color_target_reasoning = {
    "E1": (
        "The goal is to move the red block onto the magenta square. This requires a low, planar push with the wrist face, maintaining continuous side contact and translating the block toward the target square direction (rightward), then stopping when the block is centered on the magenta square before retracting. "
        "Action Plan <0> places the end-effector on the right face of the block, so any forward push drives the block leftward, away from the magenta target and toward the green square; this violates the required push direction. "
        "Action Plan <1> places the end-effector on the left face, aligns the push direction with the line from the block to the magenta square, maintains low planar contact, and advances the block rightward toward the target; this matches the needed motion. "
        "Action Plan <2> applies contact from above/back, which tends to induce downward or diagonal motion toward the yellow square and does not align with the needed rightward slide. "
        "Among the three trajectories, Action Plan <1> alone satisfies the contact geometry and push direction and most efficiently achieves the goal."
    )
}

E_slide_block_to_color_target_scene_description = {
    "E1": (
        "Initial scene: a tabletop with colored markers and blocks. A green square sits at left-middle. "
        "A red block (cube) is near center-left, a blue square is upper-left, a magenta square is mid-right, and a yellow square is lower-right. The robot wrist is above the center column with the gripper open and pitched downward."
    )
}

E_slide_block_to_color_target_traj_description = {
    "E1": [
        "Candidate Action Plan <0>: the end effector approaches from the top-right, makes side contact on the cube’s right (east) face, pushes leftward toward the green square, and ends with the cube and gripper closest to the green square.",
        "Candidate Action Plan <1>: the end effector descends at the center-left, makes side contact on the cube’s left (west) face, pushes rightward toward the magenta square, and ends with the cube and gripper closest to the magenta square.",
        "Candidate Action Plan <2>: the end effector lowers from above, contacts the far face of the cube to push it, and ends with the cube and gripper closest to the yellow square.",
    ]
}

E_slide_block_to_color_target_reasoning = {
    "E1": (
        "The goal is to move the red block onto the magenta square. This requires a planar push with the wrist face, maintaining continuous side contact while translating the block rightward toward the target square, then stopping once the block is centered on the magenta square before retracting. "
        "Action Plan <0> pushes the red block toward the green square, which violates the target color. "
        "Action Plan <1> pushes from the left face, aligning with the correct rightward direction and ending on the magenta square — this matches the goal. "
        "Action Plan <2> pushes from above, resulting in the block ending near the yellow square, which indicates it was pushed in the wrong direction and to the wrong target color. "
        "Therefore, only Action Plan <1> satisfies both the push direction and the final target color requirements, and it most efficiently achieves the goal."
    )
}

E_slide_block_to_color_target_best_plan = {"E1": 1}


# =========================
# Insert onto square peg
# =========================
E_insert_onto_square_peg_task_goal = {"E1": "Put the blue ring on the red spoke."}

E_insert_onto_square_peg_scene_description = {
    "E1": (
        "Initial scene: a tabletop with a brown base holding three vertical spokes arranged left-to-right: red (left), pink (middle), and purple (right). "
        "A blue square ring lies on the table in front of the base. The robot wrist is above the workspace with the gripper open."
    )
}

E_insert_onto_square_peg_traj_description = {
    "E1": [
        "Candidate Action Plan <0>: the end-effector grasps the blue ring and its final position is closest to the left side of the base where the red spoke stands.",
        "Candidate Action Plan <1>: the end-effector grasps the blue ring, lifts it, and moves closest to the middle pink spoke.",
        "Candidate Action Plan <2>: the end-effector keeps the gripper closed without the ring and moves closest to the left red spoke.",
    ]
}

E_insert_onto_square_peg_reasoning = {
    "E1": (
        "The goal is to insert the blue ring onto the red spoke. To achieve this, we need to (1) acquire the ring with a stable top-down grasp and lift it from the table, "
        "(2) translate and orient the ring so that its hole is coaxial with the left red spoke, (3) descend along the red spoke until the ring is fully seated, and "
        "(4) release and retract while maintaining vertical alignment. "
        "Action Plan <0> satisfies step (1) by grasping and lifting the ring and partially satisfies step (2) because the ring is positioned closest to the red spoke, making the remaining vertical insertion short and well aligned. "
        "Action Plan <1> also satisfies step (1) but violates step (2) by moving toward the pink spoke, which is not the specified target; insertion there would fail the task. "
        "Action Plan <2> violates step (1) because the gripper is closed without holding the ring; although it is near the red spoke, it cannot perform steps (2)–(4) without first picking up the ring. "
        "Among the three trajectories, Action Plan <0> alone both holds the ring and approaches the correct red spoke, requiring only a short, well-aligned insertion to complete the task, so it is the best choice."
    )
}

E_insert_onto_square_peg_best_plan = {"E1": 0}


# =========================
# Stack cups
# =========================
E_stack_cups_task_goal = {"E1": "Stack the other cups on top of the green cup."}

E_stack_cups_scene_description = {
    "E1": (
        "Initial scene: a tabletop with three upright cups near the front-left corner: the green cup at front-left (target base), the blue cup to its right-front, and the black cup slightly behind the green on the left-rear. "
        "The robot wrist is low over the cups with the parallel-jaw gripper open and centered between the green and blue cups."
    )
}

E_stack_cups_traj_description = {
    "E1": [
        "Candidate Action Plan <0>: the end-effector grasps the blue cup near its rim. Its final position is closest to the green cup.",
        "Candidate Action Plan <1>: the end-effector grasps the black cup near its rim. Its final position is closest to the green cup.",
        "Candidate Action Plan <2>: the end-effector grasps the green cup near its rim. Its final position is closest to the black cup.",
    ]
}

E_stack_cups_reasoning = {
    "E1": (
        "The goal is to place the non-green cups onto the green cup to form a stable stack. To achieve this, we need to (1) acquire a non-green cup with a stable top-down rim grasp, "
        "(2) translate to the green cup, (3) align the cup axes coaxially with minimal tilt, (4) descend until the green cup captures the lifted cup, and (5) release and repeat for the remaining non-green cup. "
        "Action Plan <0> satisfies steps (1) and (2) for the blue cup by grasping it and moving close to the green cup; the plan sets up steps (3)–(5) and is valid. "
        "Action Plan <1> also satisfies steps (1) and (2) for the black cup and places it even closer to the green cup, which reduces the remaining alignment and insertion travel, so it is slightly more efficient for the first stacking operation. "
        "Action Plan <2> violates the task decomposition because it grasps the base (green) cup; moving the base prevents stacking and risks destabilizing the scene. "
        "Among the three trajectories, Action Plan <1> positions a non-green cup closest to the base while maintaining a secure grasp, thus minimizing the remaining steps and time to complete the stack. "
        "Therefore, Action Plan <1> is the best choice."
    )
}

E_stack_cups_best_plan = {"E1": 1}
# =========================
# Helpers + Unified builder (once)
# =========================
_ids = ["E1"]  # extend if adding E2, E3, ...

def compose_visual_state(scene_dict, traj_dict, ex_id=None) -> str:
    """Join scene description with candidate plans; preserves original spacing between sentences."""
    if ex_id is None:
        return scene_dict + " " + " ".join(traj_dict)
    else:
        return scene_dict[ex_id] + " " + " ".join(traj_dict[ex_id])

def _compose_evaluator_json(task_goal: str, visual: str, reasoning: str, best_plan: int) -> str:
    """
    Create the JSON string block for the evaluator task.
    Returns a string with input and output sections, formatted for clarity.
    """
    input_section = (
        "Input:\n"
        f'Visual Descriptions of Candidate Action Plans: "{visual}",\n'
    )
    output_section = (
        "Output:\n"
        "{\n"
        f'"task_goal": "{task_goal}",\n'
        f'"reasoning": "{reasoning}",\n'
        f'"current_best_plan": {best_plan}\n'
        "}"
    )
    return " [" + input_section + output_section + "] "

def _compose_descriptor_json(scene_description: str, traj_description: str) -> str:
    """Create the JSON string block for the descriptor task."""
    return (
        "{\n"
        f'  "scene_description": "{scene_description}",\n'
        f'  "action_trajectory_description": "{traj_description}",\n'
        "}"
    )

def build_task_examples(scene_dict, traj_dict, task_goal_dict, reasoning_dict, best_plan_dict, ids):
    """Return a dict with 'evaluator' and 'descriptor' lists for a task."""
    evaluator = [
        _compose_evaluator_json(
            task_goal_dict[i],
            compose_visual_state(scene_dict, traj_dict, i),
            reasoning_dict[i],
            best_plan_dict[i],
        )
        for i in ids
    ]
    # bind i first, then j over that i's trajectories
    # replace the pattern "Candidate Action Plan\s*<\s*(\d+)\s*>: " with "" in each description
    descriptor = []
    for i in ids:
        for j in range(len(traj_dict[i])):
            pattern = f"Candidate Action Plan <{j}>: "
            if pattern not in traj_dict[i][j]:
                raise ValueError(f"Pattern '{pattern}' not found in trajectory description: {traj_dict[i][j]}")
            desc = traj_dict[i][j].replace(pattern, "")
            descriptor.append(_compose_descriptor_json(scene_dict[i], desc))
    return {"evaluator": evaluator, "descriptor": descriptor}

# =========================
# Task registry (no eval)
# =========================
TASK_SPECS = {
    "push_buttons": {
        "scene":      E_push_buttons_scene_description,
        "traj":       E_push_buttons_traj_description,
        "goal":       E_push_buttons_task_goal,
        "reasoning":  E_push_buttons_reasoning,
        "best_plan":  E_push_buttons_best_plan,
    },
    "slide_block_to_color_target": {
        "scene":      E_slide_block_to_color_target_scene_description,
        "traj":       E_slide_block_to_color_target_traj_description,
        "goal":       E_slide_block_to_color_target_task_goal,
        "reasoning":  E_slide_block_to_color_target_reasoning,
        "best_plan":  E_slide_block_to_color_target_best_plan,
    },
    "insert_onto_square_peg": {
        "scene":      E_insert_onto_square_peg_scene_description,
        "traj":       E_insert_onto_square_peg_traj_description,
        "goal":       E_insert_onto_square_peg_task_goal,
        "reasoning":  E_insert_onto_square_peg_reasoning,
        "best_plan":  E_insert_onto_square_peg_best_plan,
    },
    "stack_cups": {
        "scene":      E_stack_cups_scene_description,
        "traj":       E_stack_cups_traj_description,
        "goal":       E_stack_cups_task_goal,
        "reasoning":  E_stack_cups_reasoning,
        "best_plan":  E_stack_cups_best_plan,
    },
}

# =========================
# Build once for all tasks
# =========================
genex_vlm_few_shot_examples = {
    name: build_task_examples(
        spec["scene"], spec["traj"], spec["goal"], spec["reasoning"], spec["best_plan"], _ids
    )
    for name, spec in TASK_SPECS.items()
}

