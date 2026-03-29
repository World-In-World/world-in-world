from wiw_manip.planner.openpi_planner import OpenpiPlanner
from wiw_manip.planner.action_igenex_planner import ActionIgenexPlanner, select_diverse_points

class OpenpiIgenexPlanner(OpenpiPlanner, ActionIgenexPlanner):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shot_example_num = 1

    def act(self,
            img_path_list_anno,
            user_instruction,
            avg_obj_coord,
            task_variation,
            img_path_list_origin,
            curr_obs,
            last_act,
            **kwargs,
        ):
        proposer_fn = self.generate_trajectories(
            curr_obs=curr_obs, user_instruction=user_instruction
        )

        return self.query_igenex(img_path_list_anno, user_instruction, img_path_list_origin, curr_obs, proposer_fn, self.task_name, **kwargs)

    def generate_trajectories(
        self, curr_obs, user_instruction
    ):
        def proposer_fn(num_trajs, accumulate_trajs=[]):
            actions_plans, reasons = OpenpiPlanner.act(
                self=self,
                curr_obs=curr_obs,
                user_instruction=user_instruction,
                query_num=self.proposal_num,
            )

            all_trajs = actions_plans + accumulate_trajs
            end_points = [trajectory[-1][:3] for trajectory in all_trajs]
            indexies = select_diverse_points(end_points, num_trajs)
            selected_trajs = [all_trajs[i] for i in indexies]

            return selected_trajs

        return proposer_fn