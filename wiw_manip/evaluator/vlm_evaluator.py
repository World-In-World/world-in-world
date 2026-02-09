from wiw_manip.envs.RLBenchEnv import RLBenchEnv, VALID_TASKS
from wiw_manip.evaluator.config.eb_manipulation_example import vlm_examples_RLbench
from wiw_manip.main import logger
from wiw_manip.envs.eb_man_utils import get_continous_action_from_discrete
from wiw_manip.evaluator.base_evaluator import Base_Evaluator

class VLM_Evaluator(Base_Evaluator):
    def __init__(self, config):
        self.model_name = config['model_name']
        self.config = config
        self.env = None
        self.planner = None

    def env_reset(self):
        return self.env.reset()

    def env_step(self, action):
        action = get_continous_action_from_discrete(action)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def evaluate_main(self, tasks=VALID_TASKS):
        for i, eval_set in enumerate(tasks):
            self.eval_task = eval_set
            logger.info(f'Current eval set: {eval_set}')
            if "/" in self.model_name:
                real_model_name = self.model_name.split('/')[1]
            else:
                real_model_name = self.model_name
            if 'exp_name' not in self.config or self.config['exp_name'] is None:
                self.log_path = "running/{}/{}/n_shot={}_resolution={}_detection_box={}_multiview={}_multistep={}_visual_icl={}/{}".format(
                    self.config.env,
                    real_model_name,
                    self.config["n_shots"], self.config["resolution"],
                    self.config["detection_box"], self.config["multiview"],
                    self.config["multistep"], self.config["visual_icl"],
                    self.eval_task,
                )
            else:
                self.log_path = "running/{}/{}/{}/{}".format(
                    self.config.env, real_model_name, self.config["exp_name"], self.eval_task
                )

            if i == 0:
                self.env = RLBenchEnv(
                    eval_set=self.eval_task,
                    img_size=(self.config["resolution"], self.config["resolution"]),
                    down_sample_ratio=self.config["down_sample_ratio"],
                    log_path=self.log_path,
                    enable_path_obs=self.config["enable_path_obs"],
                    exp_name=self.config.get("exp_name", None),
                    max_step=self.config["max_step"],
                )
            else:
                self.env.init_dataset_and_tasks(
                    eval_task=self.eval_task,
                    down_sample_ratio=self.config["down_sample_ratio"],
                    log_path=self.log_path,
                )
            ic_examples = self.load_demonstration()
            self.initialize_planner(ic_examples, self.eval_task)
            self.evaluate()

    def load_demonstration(self):
        all_examples = {}
        for task in VALID_TASKS:
            all_examples[task] = vlm_examples_RLbench.get(task.split('_')[0], None)

        return all_examples