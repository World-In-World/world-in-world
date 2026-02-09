from wiw_manip.evaluator.config.system_prompts import (
    eb_manipulation_system_prompt,
    genex_revise_manipulation_auxiliary_prompt,
)
from wiw_manip.planner.igenex_planner import IgenexPlanner
from wiw_manip.evaluator.vlm_evaluator import VLM_Evaluator

class Igenex_Evaluator(VLM_Evaluator):
    def __init__(self, config):
        super().__init__(config)
        self.igenex_host = config["igenex_host"]
        self.proposal_num = config["proposal_num"]
        self.mpc_mode = config["mpc_mode"]

    def initialize_planner(self, ic_examples, task_name):
        self.planner = IgenexPlanner(
            model_name=self.model_name,
            model_type=self.config["model_type"],
            system_prompt=eb_manipulation_system_prompt,
            revise_aux_prompt=genex_revise_manipulation_auxiliary_prompt,
            examples=ic_examples,
            n_shot=self.config["n_shots"],
            chat_history=self.config["chat_history"],
            language_only=self.config["language_only"],
            multiview=self.config["multiview"],
            multistep=self.config["multistep"],
            visual_icl=self.config["visual_icl"],
            tp=self.config["tp"],
            executed_action_per_step=self.config["executed_action_per_step"],
            proposal_num=self.proposal_num,
            igenex_host=self.igenex_host,
            mpc_mode=self.mpc_mode,
            vlm_args={
                "temperature": self.config.get("vlm__temperature"),
                "top_k": self.config.get("vlm__top_k"),
            },
            pred_img_size=self.config["pred_img_size"],
            exp_name=self.config.get("exp_name", None),
        )
