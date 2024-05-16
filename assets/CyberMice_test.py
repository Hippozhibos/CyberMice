"""Tests for the CyberMice"""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control import viewer
from dm_control.composer.observation.observable import base as observable_base
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import rodent

from CyberMice import Mice
import numpy as np

_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.001


def walker_run(random_state=None):
    walker = Mice()
    arena = corr_arenas.EmptyCorridor()
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0, 0),
        walker_spawn_rotation=0,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)

viewer.launch(environment_loader=walker_run)

import numpy as np
env = walker_run()

# Get the `action_spec` describing the control inputs.
action_spec = env.action_spec()

# Step through the environment for one episode with random actions.
time_step = env.reset()
while not time_step.last():
    action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
    time_step = env.step(action)
    print("reward = {}, discount = {}, observations = {}.".format(
    time_step.reward, time_step.discount, time_step.observation))