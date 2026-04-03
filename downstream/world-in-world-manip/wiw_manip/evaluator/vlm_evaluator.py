import numpy as np

from wiw_manip.envs.RLBenchEnv import RLBenchEnv
from wiw_manip.envs.LiberoEnv import LiberoEnv
from wiw_manip.evaluator.config.eb_manipulation_example import vlm_examples_RLbench
from wiw_manip.main import logger
from wiw_manip.envs.utils import (
    RLBENCH_VALID_TASKS,
    LIBERO_VALID_TASKS,
    describe_discrete_action_conversion,
    get_continous_action_from_discrete,
)
from wiw_manip.evaluator.base_evaluator import Base_Evaluator

class VLM_Evaluator(Base_Evaluator):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = config['model_name']

    def env_reset(self):
        return self.env.reset()

    def env_step(self, action):
        if self.backend == "libero":
            discrete_action = [int(x) for x in action]
            # action, debug_payload = describe_discrete_action_conversion(
            action = describe_discrete_action_conversion(
                discrete_action,
                bounds=self.env.scene_bounds,
                flip_y=False,
                zero_rotation_as_identity_wxyz=False,
            )
            if len(discrete_action) == 7 and discrete_action[3:6] == [0, 0, 0]:
                curr_quat = np.asarray(self.env.last_frame_obs["robot0_eef_quat"], dtype=np.float64)
                quat_norm = np.linalg.norm(curr_quat)
                if quat_norm > 1e-8:
                    curr_quat = curr_quat / quat_norm
                    action[3:7] = curr_quat
                    # debug_payload["rotation_override"] = (
                    #     "all-zero-rpy -> use current eef quaternion from env reset/last step"
                    # )
                    # debug_payload["target_quaternion_xyzw"] = curr_quat.astype(float).tolist()
            # obs, reward, done, info = self.env.step(action, debug_payload=debug_payload)
            obs, reward, done, info = self.env.step(action)
        elif self.backend == "rlbench":
            action = get_continous_action_from_discrete(
                action,
                bounds=self.env.scene_bounds,
                flip_y=False,
            )
            obs, reward, done, info = self.env.step(action)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        return obs, reward, done, info

    def _get_env_class(self):
        if self.backend == "libero":
            return LiberoEnv
        return RLBenchEnv

    def _get_eval_tasks(self):
        if self.backend == "libero":
            return LIBERO_VALID_TASKS
        return RLBENCH_VALID_TASKS

    @staticmethod
    def _normalize_eval_sets(eval_sets):
        if eval_sets is None:
            return []
        if isinstance(eval_sets, str):
            return [eval_sets] if eval_sets else []
        return [str(item) for item in eval_sets if str(item)]

    def _resolve_eval_tasks(self, tasks=None):
        backend = self.backend
        if tasks is not None:
            resolved = [str(item) for item in tasks]
        else:
            configured = self._normalize_eval_sets(self.config.get("eval_sets", None))
            if len(configured) > 0:
                resolved = configured
            elif backend == "libero":
                # Keep backward-compatible default behavior for LIBERO.
                resolved = ["libero_object"]
            else:
                resolved = list(self._get_eval_tasks())

        valid = set(self._get_eval_tasks())
        invalid = [task for task in resolved if task not in valid]
        if len(invalid) > 0:
            raise ValueError(
                f"Unsupported eval_sets for backend '{backend}': {invalid}. "
                f"Supported: {sorted(valid)}"
            )
        return resolved

    def evaluate_main(self, tasks=None):
        tasks = self._resolve_eval_tasks(tasks)
        env_class = self._get_env_class()
        backend = self.backend
        logger.info("Resolved eval tasks: %s", tasks)
        for i, eval_set in enumerate(tasks):
            self.eval_task = eval_set
            logger.info(f'Current eval set: {eval_set}')
            if "/" in self.model_name:
                real_model_name = self.model_name.split('/')[1]
            else:
                real_model_name = self.model_name
            if 'exp_name' not in self.config or self.config['exp_name'] is None:
                self.log_path = "running/{}/{}/{}/n_shot={}_resolution={}_detection_box={}_multiview={}_multistep={}_visual_icl={}/{}".format(
                    self.config.env,
                    backend,
                    real_model_name,
                    self.config["n_shots"], self.config["resolution"],
                    self.config["detection_box"], self.config["multiview"],
                    self.config["multistep"], self.config["visual_icl"],
                    self.eval_task,
                )
            else:
                self.log_path = "running/{}/{}/{}/{}/{}".format(
                    self.config.env, backend, real_model_name, self.config["exp_name"], self.eval_task
                )

            if i == 0:
                env_kwargs = {}
                if backend == "libero":
                    env_kwargs["server_url"] = self.config.get("libero_env_url")
                    env_kwargs["action_repeat"] = int(self.config.get("libero_action_repeat", 50))
                self.env = env_class(
                    eval_set=self.eval_task,
                    img_size=(self.config["resolution"], self.config["resolution"]),
                    down_sample_ratio=self.config["down_sample_ratio"],
                    log_path=self.log_path,
                    enable_path_obs=self.config["enable_path_obs"],
                    exp_name=self.config.get("exp_name", None),
                    max_step=self.config["max_step"],
                    **env_kwargs,
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
        if self.backend == "libero":
            for task in LIBERO_VALID_TASKS:
                all_examples[task] = vlm_examples_RLbench.get(task, None)
            return all_examples

        for task in RLBENCH_VALID_TASKS:
            all_examples[task] = vlm_examples_RLbench.get(task.split('_')[0], None)

        return all_examples
