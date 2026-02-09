import pandas as pd
from typing import List, Dict, Callable
try:
    from tabulate import tabulate          # pip install tabulate
except ImportError:
    tabulate = None


class State:
    """
    Class to store the state trajectory of an agent.
    It wraps a pandas DataFrame for state rows (e.g. position, rotation, etc.),
    provides lists for actions/answers, and tracks the best recognized answer.
    """

    def __init__(self, columns=None):
        self.state_traj = pd.DataFrame(columns=columns or [])
        self.action_traj = []
        self.answer_traj = []
        self.answer_val_traj = []
        self.pending_actions = []
        self._current_state_obs = None

        # new attributes for best recognized answer:
        self._best_answer = None
        self._best_answer_val = 0.0

        self._position_traj = []  # calc path len in meters

    def __len__(self):
        return len(self.state_traj)

    def save_state(self, save_path: str):
        """
        Save the state trajectory to a CSV file.
        """
        self.state_traj.to_csv(save_path, index=False)
        print(f"State trajectory saved to: {save_path}")

    def __repr__(self) -> str:
        """
        Return a human-readable snapshot of ``state_traj``.

        Uses a GitHub-flavoured table when the *tabulate* package is
        available; otherwise falls back to ``DataFrame.to_string()``.

        *Shows the **entire** DataFrame.*  If your trajectories can grow
        large and you’d rather cap the output, change the ``df`` slice
        below (e.g. ``df = df.tail(20)``).
        """
        if self.state_traj.empty:
            return "(state trajectory is empty)"

        df = self.state_traj
        # create a copy of df filled *all* missing values with None for tabulate
        clean_df = df.astype(object).where(pd.notna(df), None)

        if tabulate is not None:
            return tabulate(clean_df, headers="keys", tablefmt="github", showindex=True)
        else:
            # Fallback: still readable, just not as nicely formatted
            return df.to_string()

    # Let ``str(state)`` (and therefore ``print(state)``) reuse the same text
    __str__ = __repr__

    @property
    def position_traj(self):
        return self._position_traj

    def update_position_traj(self, new_position):
        self._position_traj.append(new_position)

    def add_new_state(self, new_state_dict: Dict[str, str], state_imgs: Dict[str, Callable] = None):
        new_row = pd.DataFrame([new_state_dict])
        self.state_traj = pd.concat([self.state_traj, new_row], ignore_index=True)
        self._current_state_obs = state_imgs

    def record_past_action(self, new_action):
        self.action_traj.append(new_action)

    def add_pending_actions(self, actions: List[str]):
        assert len(self.pending_actions) == 0
        self.pending_actions.extend(actions)

    def add_answer(self, ans, ans_val=None):
        self.answer_traj.append(ans)
        self.answer_val_traj.append(ans_val)

    def add_to_recent_state(self, pred_save_paths: List[str], key: str, mode="replace"):
        """
        Update the *latest row* of ``self.state_traj`` at column ``key``.
        mode : {'replace', 'extend'}, default 'replace'
            - replace – overwrite the existing entry with ``pred_save_paths``
            - extend  – extend the existing list with ``pred_save_paths``
        """
        if not isinstance(pred_save_paths, list):
            pred_save_paths = [pred_save_paths]
        if len(pred_save_paths) > 0:
            print(f"Len of pred_save_paths to assist decision: {len(pred_save_paths)}:\t")
        if key not in self.state_traj.columns:
            self.state_traj[key] = None

        if mode == "replace":
            self.state_traj.at[self.state_traj.index[-1], key] = pred_save_paths
        elif mode == "extend":
            existing = self.state_traj.at[self.state_traj.index[-1], key]
            # If existing is None or a single item, normalise to a list
            if not isinstance(existing, list) and pd.isna(existing):
                existing = []
            existing.extend(pred_save_paths)
            self.state_traj.at[self.state_traj.index[-1], key] = existing
        else:
            raise ValueError(f"Unknown mode '{mode}' for add_to_recent_state. Use 'replace' or 'extend'.")

    def is_empty(self) -> bool:
        return self.state_traj.empty

    # -------- New helper methods --------
    def fetch_current_state_obs(self, key: str):
        """
        Returns the current state observation.
        """
        return self._current_state_obs[key]

    def pop_next_pending_action(self):
        """
        Pops the next action from the pending actions list.
        """
        if self.pending_actions:
            return self.pending_actions.pop(0)
        else:
            return None

    def get_pending_action_num(self):
        """
        return the remaining length of pending actions.
        """
        return len(self.pending_actions)

    def get_from_history(self, key: str) -> List:
        """
        Fetches all the values from the specified state column, and filter out empty values.
        """
        if key not in self.state_traj.columns.tolist():
            return []
        else:
            not_empty_items = []
            for val in self.state_traj[key].tolist():
                if isinstance(val, list):
                    if pd.notna(val).all():
                        not_empty_items.append(val)
                elif pd.notna(val):
                    not_empty_items.append(val)
            return not_empty_items

    def get_from_recent_state(self, key: str) -> List:
        """
        get the value of the key from the most recent state.
        """
        if key not in self.state_traj.columns.tolist():
            return None
        else:
            val = self.state_traj[key].tolist()[-1]
            if isinstance(val, list):
                if pd.notna(val).all():
                    return val
            elif pd.notna(val):
                return val
            else:
                return None

    def clean_up_history(self, key: str):
        """
        Cleans up the history for the specified key by set all the values to pd none in the column.
        """
        if key in self.state_traj.columns.tolist():
            self.state_traj[key] = pd.Series([pd.NA] * len(self.state_traj))
        else:
            print(f"WARNING: {key} not in state trajectory columns.")

    def get_all_recorded_keys(self):
        """
        Returns all the keys recorded in the state trajectory.
        """
        return self.state_traj.columns.tolist()

    def get_best_answer(self):
        return self._best_answer

    def set_best_answer(self, ans):
        self._best_answer = ans

    def get_best_answer_val(self):
        return self._best_answer_val

    def set_best_answer_val(self, val):
        self._best_answer_val = val

    def get_state_traj(self):
        """
        Returns the entire state DataFrame if needed elsewhere.
        """
        return self.state_traj

    def get_action_traj(self):
        """
        Returns the action trajectory list.
        """
        return self.action_traj
