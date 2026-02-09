from typing import List
import numpy as np
from pyrep.objects import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from amsolver.backend.conditions import DetectedCondition, ConditionSet
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task

colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('maroon', (0.5, 0.0, 0.0)),
    ('lime', (0.0, 1.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]

class InsertOntoSquarePeg(Task):

    def init_task(self) -> None:
        self._square_ring = Shape('square_ring')
        self._success_centre = Dummy('success_centre')
        success_detectors = [ProximitySensor(
            'success_detector%d' % i) for i in range(4)]
        self.register_graspable_objects([self._square_ring])
        success_condition = ConditionSet([DetectedCondition(
            self._square_ring, sd) for sd in success_detectors])
        self.register_success_conditions([success_condition])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        spokes = [Shape('pillar0'), Shape('pillar1'), Shape('pillar2')]
        chosen_pillar = np.random.choice(spokes)
        chosen_pillar.set_color(color_rgb)
        _, _, z = self._success_centre.get_position()
        x, y, _ = chosen_pillar.get_position()
        self._success_centre.set_position([x, y, z])

        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        spokes.remove(chosen_pillar)
        for spoke, i in zip(spokes, color_choices):
            name, rgb = colors[i]
            spoke.set_color(rgb)
        b = SpawnBoundary([Shape('boundary0')])
        b.sample(self._square_ring)
        return ['put the ring on the %s spoke' % color_name,
                'slide the ring onto the %s colored spoke' % color_name,
                'place the ring onto the %s spoke' % color_name]

    def variation_count(self) -> int:
        return len(colors)
