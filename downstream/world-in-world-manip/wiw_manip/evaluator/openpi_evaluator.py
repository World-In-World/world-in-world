from wiw_manip.evaluator.vlm_evaluator import VLM_Evaluator
from wiw_manip.planner.openpi_planner import OpenpiPlanner
import numpy as np

class Openpi_Evaluator(VLM_Evaluator):
        
    def initialize_planner(self, ic_examples, task_name):
        self.planner = OpenpiPlanner(
            backend=self.backend,
            task=task_name,
        )
        
    def act(self, img_path_list, image_history, user_instruction, avg_obj_coord, obs, img_path_list_origin=None):
        return self.planner.act(curr_obs=obs, user_instruction=user_instruction)
    
    def env_reset(self):
        return super().env_reset()

    def env_step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info