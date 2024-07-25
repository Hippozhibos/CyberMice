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
"""Tasks for multi-clip mocap tracking with RL."""

import abc
import h5py

import typing
from typing import Any, Callable, Mapping, Optional, Sequence, Set, Text, Union

from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.variation import noises
from dm_control.composer.variation import distributions

from dm_control.locomotion.tasks.reference_pose import rewards
from dm_control.locomotion.arenas import floors

from dm_control.mujoco.wrapper import mjbindings

from assets.CyberMice import Mice
from tasks.rewards import (get_reference_features, get_walker_features,
                                   reward_factors_deep_mimic)

import numpy as np


mjlib = mjbindings.mjlib
DEFAULT_PHYSICS_TIMESTEP = 0.005
_MAX_END_STEP = 10000

class DotsTracking(composer.Task, metaclass=abc.ABCMeta):
    
    def __init__(
            self, 
            walker,
            ref_path,
            reward_type: Text = 'termination_reward'
            ):
        # Initialize any necessary attributes for the MiceTracking task.
        self._walker = walker
        self._ref_path = ref_path
        self._arena = floors.Floor()
        self._arena.add_free_entity(self._walker)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

        # Configure and enable observables
        pos_corrptor = noises.Additive(distributions.Normal(scale=0.01))
        self._walker.observables.joint_positions.corruptor = pos_corrptor
        self._walker.observables.joint_positions.enabled = True
        vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.01))
        self._walker.observables.joint_velocities.corruptor = vel_corruptor
        self._walker.observables.joint_velocities.enabled = True

        self._task_observables = {}

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Load the mocap data
        self.mocap_data = self._load_mocap_data(self._ref_path)

        # Define reward-related variables  
        self._reward_fn = rewards.get_reward(reward_type)
        self._reward_keys = rewards.get_reward_channels(reward_type)

    def _load_mocap_data(self, file_path):
        with h5py.File(file_path, 'r') as f:
            qpos = f['qpos'][:]
            qvel = f['qvel'][:]
        return {'qpos_ref': qpos, 'qvel_ref': qvel}

    def initialize_episode_mjcf(self, random_state):
        """Modifies the MJCF model of this task before the next episode begins.

        Args:
          random_state: An instance of `np.random.RandomState`.
        """
        # Implement any task-specific MJCF initialization
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        """Modifies the physics state before the next episode begins.

        Args:
          physics: An instance of `control.Physics`.
          random_state: An instance of `np.random.RandomState`.
        """
        # Implement any task-specific physics initialization
        # Randomly select a frame
        num_frames = self.mocap_data['qpos_ref'].shape[0]
        frame_idx = random_state.randint(num_frames)

        # Set the initial qpos and qvel
        qpos_ref = self.mocap_data['qpos_ref'][frame_idx]
        qvel_ref = self.mocap_data['qvel_ref'][frame_idx]

        # physics.data.qpos[7:] = qpos
        print(qvel_ref.shape)
        physics.data.qpos[1:-6] = qpos_ref[6:]
        physics.data.qpos[-6:] = qpos_ref[:6]
        physics.data.qvel[:] = qvel_ref

    def before_step(self, physics, action, random_state):
        """A callback which is executed before an agent control step.

        Args:
          physics: An instance of `control.Physics`.
          action: A NumPy array corresponding to agent actions.
          random_state: An instance of `np.random.RandomState` (unused).
        """
        super().before_step(physics, action, random_state)
        # Implement any additional logic before each control step
        pass

    def after_step(self, physics, random_state):
        """A callback which is executed after an agent control step.

        Args:
          physics: An instance of `control.Physics`.
          random_state: An instance of `np.random.RandomState`.
        """
        # Implement any additional logic after each control step
        pass

    def get_reward(self, physics):
        """Returns factorized reward terms."""
        step = round(physics.time() / self.control_timestep)
        walker_ft = get_walker_features(physics, self._mocap_joints,
                                        self._mocap_sites)
        reference_ft = get_reference_features(self._snippet, step)
        reward_factors = reward_factors_deep_mimic(
            walker_features=walker_ft,
            reference_features=reference_ft,
            weights=(20, 1, 1, 1))
        return reward_factors


    def should_terminate_episode(self, physics):
        """Determines whether the episode should terminate given the physics state.

        Args:
          physics: A Physics object.

        Returns:
          A boolean indicating whether to terminate the episode.
        """
        # Implement logic to determine if the episode should terminate
        return False

    def get_discount(self, physics):
        """Calculates the reward discount factor given the physics state.

        Args:
          physics: A Physics object.

        Returns:
          A float representing the discount factor.
        """
        # Implement logic to calculate the discount factor
        return 1.0

    @property
    def task_observables(self):
        """An OrderedDict of task-specific `control.Observable` instances.

        Returns:
          An `collections.OrderedDict` mapping strings to instances of
          `control.Observable`.
        """
        return self._task_observables

    @property
    def root_entity(self):
        """A `base.Entity` instance for this task."""
        return self._arena
    
    @property
    def walker(self):
        return self._walker