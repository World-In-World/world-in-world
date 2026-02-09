import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import yaml
from wiw_manip.envs.eb_man_utils import VALID_TASKS, DIFF_VALID_TASKS

logger = logging.getLogger("EB_logger")
if not logger.hasHandlers():
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

# the corresponding evaluator class
class_names = {
    "eb-man": "EB_ManipulationEvaluator",
    "diff-base": "Diff_Evaluator",
    "diff-igenex": "Diff_Igenex_Evaluator",
    "vlm-base": "VLM_Evaluator",
    "vlm-igenex": "Igenex_Evaluator",
}

# the evaluator file you want to use
module_names = {
    "eb-man": "eb_manipulation_evaluator",
    "diff-base": "diff_evaluator",
    "diff-igenex": "diff_igenex_evaluator",
    "vlm-base": "vlm_evaluator",
    "vlm-igenex": "igenex_evaluator",
}

def get_evaluator(env_name: str):

    if env_name not in module_names:
        raise ValueError(f"Unknown environment: {env_name}")

    module_name = f"wiw_manip.evaluator.{module_names[env_name]}"
    evaluator_name = class_names[env_name]

    module = __import__(module_name, fromlist=[evaluator_name])
    return getattr(module, evaluator_name)

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.getLogger().handlers.clear()
    if 'log_level' not in cfg or cfg.log_level == "INFO":
        logger.setLevel(logging.INFO)
    if 'log_level' in cfg and cfg.log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    env_name = cfg.env
    logger.info(f"Evaluating environment: {env_name}")
    config = cfg

    logger.info(f"Config: {OmegaConf.to_yaml(config)}")
    logger.info("Starting evaluation")
    evaluator_class = get_evaluator(env_name)
    evaluator = evaluator_class(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main(VALID_TASKS if 'diff' not in env_name else DIFF_VALID_TASKS)
    logger.info("Evaluation completed")

if __name__ == "__main__":
    main()