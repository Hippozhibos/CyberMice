# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Produces reference environments for CMU humanoid tracking task."""

import sys
sys.path.append('D:/Mujoco/CyberMice')  # Adjust the path as needed
import PIL.Image
import numpy as np
from typing import Callable, Union

from dm_control import composer
from dm_control.locomotion import arenas

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import tracking

from dm_control.locomotion.walkers import cmu_humanoid

from assets.CyberMice import Mice

from tasks.walk_imitation import WalkImitation
from tasks.trajectory_loaders import (
    HDF5WalkingTrajectoryLoader,
    InferenceWalkingTrajectoryLoader,
    MiceTrajectoryLoader
)

def walk_imitation(ref_path: Union[str, None] = None,
                   random_state: Union[np.random.RandomState, None] = None,
                   terminal_com_dist: float = 0.3):
    """Requires a fruitfly to track a reference walking fly.

    Args:
        ref_path: Path to reference trajectory dataset. If not provided, task
            will run in inference mode with InferenceWalkingTrajectoryLoader,
            without loading actual walking dataset.
        random_state: Random state for reproducibility.
        terminal_com_dist: Episode will be terminated when distance from model
            CoM to ghost CoM exceeds terminal_com_dist. Can be float('inf').
    Returns:
        Environment for walking tracking task.
    """
    # Build a fruitfly walker and arena.
    walker = Mice
    arena = arenas.floors.Floor()
    # Initialize a walking trajectory loader.
    if ref_path is not None:
        inference_mode = False
        traj_generator = HDF5WalkingTrajectoryLoader(
            path=ref_path, random_state=random_state)
    else:
        inference_mode = True
        traj_generator = InferenceWalkingTrajectoryLoader()
    # Build a task that rewards the agent for tracking a walking ghost.
    time_limit = 10.0
    task = WalkImitation(walker=walker,
                         arena=arena,
                         traj_generator=traj_generator,
                         terminal_com_dist=terminal_com_dist,
                         mocap_joint_names=traj_generator.get_joint_names(),
                         mocap_site_names=traj_generator.get_site_names(),
                         inference_mode=inference_mode,
                        #  joint_filter=0.01,
                         future_steps=64,
                         time_limit=time_limit)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)

def vision_guided_flight(wpg_pattern_path: str,
                         bumps_or_trench: str = 'bumps',
                         random_state: Union[np.random.RandomState, None] = None,
                         **kwargs_arena):
    """Vision-guided flight tasks: 'bumps' and 'trench'.

    Args:
        wpg_pattern_path: Path to baseline wing beat pattern for WPG.
        bumps_or_trench: Whether to create 'bumps' or 'trench' vision task.
        random_state: Random state for reproducibility.
        kwargs_arena: kwargs to be passed on to arena.

    Returns:
        Environment for vision-guided flight task.
    """

    if bumps_or_trench == 'bumps':
        arena = SineBumps
    elif bumps_or_trench == 'trench':
        arena = SineTrench
    else:
        raise ValueError("Only 'bumps' and 'trench' terrains are supported.")
    # Build fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = arena(**kwargs_arena)
    # Initialize a wing beat pattern generator.
    wbpg = WingBeatPatternGenerator(base_pattern_path=wpg_pattern_path)
    # Build task.
    time_limit = 0.4
    task = VisionFlightImitationWBPG(walker=walker,
                                     arena=arena,
                                     wbpg=wbpg,
                                     time_limit=time_limit,
                                     joint_filter=0.,
                                     floor_contacts=True,
                                     floor_contacts_fatal=True)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def template_task(random_state: Union[np.random.RandomState, None] = None,
                  joint_filter: float = 0.01,
                  adhesion_filter: float = 0.007,
                  time_limit: float = 1.,
                  mjcb_control: Union[Callable, None] = None,
                  observables_options: Union[dict, None] = None,
                  action_corruptor: Union[Callable, None] = None):
    """Fake no-op walking task for testing.

    Args:
        random_state: Random state for reproducibility.
        joint_filter: Timescale of filter for joint actuators. 0: disabled.
        adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
        time_limit: Episode time limit.
        mjcb_control: Optional MuJoCo control callback, a callable with
            arguments (model, data). For more information, see
            https://mujoco.readthedocs.io/en/stable/APIreference/APIglobals.html#mjcb-control
        observables_options (optional): A dict of dicts of configuration options
            keyed on observable names, or a dict of configuration options, which
            will propagate those options to all observables.
        action_corruptor (optional): A callable which takes an action as an
            argument, modifies it, and returns it. An example use case for
            this is to add random noise to the action.

    Returns:
        Template walking environment.
    """
    # Build a fruitfly walker and arena.
    walker = fruitfly.FruitFly
    arena = floors.Floor()
    # Build a no-op task.
    task = TemplateTask(walker=walker,
                        arena=arena,
                        joint_filter=joint_filter,
                        adhesion_filter=adhesion_filter,
                        observables_options=observables_options,
                        action_corruptor=action_corruptor,
                        mjcb_control=mjcb_control,
                        time_limit=time_limit)
    # Reset control callback, if any.
    mujoco.set_mjcb_control(None)
    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


 

