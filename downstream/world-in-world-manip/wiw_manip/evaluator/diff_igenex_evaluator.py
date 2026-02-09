from wiw_manip.evaluator.diff_evaluator import Diff_Evaluator
from wiw_manip.evaluator.igenex_evaluator import Igenex_Evaluator
from wiw_manip.planner.diff_igenex_planner import DiffIgenexPlanner
from wiw_manip.evaluator.config.system_prompts import (
    eb_manipulation_system_prompt,
    genex_revise_manipulation_auxiliary_prompt,
)

class Diff_Igenex_Evaluator(Diff_Evaluator, Igenex_Evaluator):
        
    def initialize_planner(self, ic_examples, task_name):
        # almost same as Igenex_Evaluator
        self.planner = DiffIgenexPlanner(
            task=task_name,
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
        
    def act(self, img_path_list, image_history, user_instruction, avg_obj_coord, obs, img_path_list_origin=None):
        ret = self.planner.act(
            gripper_history=self.gripper_history,
            img_path_list_anno=img_path_list,
            user_instruction=user_instruction,
            avg_obj_coord=str(avg_obj_coord),
            task_variation=self.env.current_task_variation,
            img_path_list_origin=img_path_list_origin,
            curr_obs=obs,
            last_act=self.last_act,
        )
        return ret