from wiw_manip.evaluator.vlm_evaluator import VLM_Evaluator
from wiw_manip.planner.diff_planner import DiffPlanner

class Diff_Evaluator(VLM_Evaluator):
        
    def initialize_planner(self, ic_examples, task_name):
        self.planner = DiffPlanner(task_name)
        
    def act(self, img_path_list, image_history, user_instruction, avg_obj_coord, obs, img_path_list_origin=None):
        return self.planner.act(curr_obs=obs, gripper_history=self.gripper_history)
        
    def env_reset(self):
        _, obs = self.env.reset()
        self.gripper_history = [obs.gripper_pose]
        return _, obs

    def env_step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.gripper_history.append(obs.gripper_pose)
        return obs, reward, done, info