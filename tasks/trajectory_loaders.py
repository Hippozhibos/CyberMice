"""Reference trajectory loaders for fruit fly imitation tasks."""

from typing import Sequence, Tuple, Optional, Dict
from abc import ABC, abstractmethod

import h5py
import numpy as np


class HDF5TrajectoryLoader(ABC):
    """Base class for loading and serving trajectories from hdf5 datasets."""

    def __init__(self,
                 path: str,
                 traj_indices: Optional[Sequence[int]] = None,
                 random_state: Optional[np.random.RandomState] = None):
        """Initializes the base trajectory loader.

        Args:
            path: Path to hdf5 dataset file with reference rajectories.
            traj_indices: List of trajectory indices to use, e.g. for train/test
                splitting etc. If None, use all available trajectories.
            random_state: Random state for reproducibility.
        """

        if random_state is None:
            self._random_state = np.random.RandomState(None)
        else:
            self._random_state = random_state

        with h5py.File(path, 'r') as f:
            self._n_traj = len(f['trajectories'])
            self._timestep = f['timestep_seconds'][()]

        if traj_indices is None:
            self._traj_indices = np.arange(self._n_traj)
        else:
            self._traj_indices = traj_indices

    @property
    def timestep(self):
        """Dataset timestep duration, in seconds."""
        return self._timestep

    @property
    def num_trajectories(self):
        """Number of trajectories in dataset."""
        return self._n_traj

    @property
    def traj_indices(self):
        """Indices of trajectories to use for training/testing."""
        return self._traj_indices

    @abstractmethod
    def get_trajectory(self,
                       traj_idx: Optional[int] = None,
                       start_step: Optional[int] = None,
                       end_step: Optional[int] = None):
        """Returns a trajectory."""
        raise NotImplementedError("Subclasses should implement this.")

class HDF5WalkingTrajectoryLoader(HDF5TrajectoryLoader):
    """Loads and serves trajectories from hdf5 walking imitation dataset."""

    def __init__(
        self,
        path: str,
        traj_indices: Optional[Sequence[int]] = None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        """Initializes the walking trajectory loader.

        Args:
            path: Path to hdf5 dataset file with reference rajectories.
            traj_indices: List of trajectory indices to use, e.g. for train/test
                splitting etc. If None, use all available trajectories.
            random_state: Random state for reproducibility.
        """

        super().__init__(path, traj_indices, random_state=random_state)

        self._h5 = h5py.File(path, 'r')
        self._traj_lens = self._h5['trajectory_lengths']
        self._n_zeros = len(str(self._n_traj))

    def trajectory_len(self, traj_idx: int) -> int:
        """Returns length of trajectory with index traj_idx."""
        return self._traj_lens[traj_idx]

    def get_trajectory(
            self,
            traj_idx: Optional[int] = None,
            start_step: Optional[int] = None,
            end_step: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Returns a walking trajectory from the dataset.

        Args:
            traj_idx: Index of the desired trajectory. If None, a random
                trajectory is selected.
            start_step: Start index for the trajectory slice. If None, defaults
                to the beginning.
            end_step: End index for the trajectory slice. If None, defaults to
                the end.

        Returns:
            dict: Dictionary containing qpos, qvel, root2site, and joint_quat
                of the trajectory.
        """
        if traj_idx is None:
            traj_idx = self._random_state.choice(self._traj_indices)

        start_step = 0 if start_step is None else start_step
        end_step = self._traj_lens[traj_idx] if end_step is None else end_step

        key = str(traj_idx).zfill(self._n_zeros)
        snippet = self._h5['trajectories'][key]

        qpos = np.concatenate((snippet['root_qpos'][start_step:end_step],
                               snippet['qpos'][start_step:end_step]),
                              axis=1)
        qvel = np.concatenate((snippet['root_qvel'][start_step:end_step],
                               snippet['qvel'][start_step:end_step]),
                              axis=1)
        qpos[:, :2] -= qpos[0, :2]

        trajectory = {
            'qpos': qpos,
            'qvel': qvel,
            'root2site': snippet['root2site'][start_step:end_step],
            'joint_quat': snippet['joint_quat'][start_step:end_step]
        }

        return trajectory

    def get_site_names(self):
        """Returns snippet site names."""
        return [s.decode('utf-8') for s in self._h5['id2name']['sites']]

    def get_joint_names(self):
        """Returns snippet joint names."""
        return [s.decode('utf-8') for s in self._h5['id2name']['joints']]

class InferenceWalkingTrajectoryLoader():
    """Simple drop-in inference-time replacement for walking trajectory loader.
    
    This trajectory loader can be used for bypassing loading actual walking
    datasets and loading custom trajectories instead, e.g. at inference time.

    To use this class, create qpos and qvel for your custom trajectory and then
    set this trajectory for loading in the walking task by calling:
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    """

    def __init__(self):
        # Nothing here!
        pass

    def set_next_trajectory(self, qpos: np.ndarray, qvel: np.ndarray):
        """Set new trajectory to be returned by get_trajectory.
        
        Args:
            qpos: Center-of-mass trajectory, (time, 7).
            qvel: Velocity of CoM trajectory, (time, 6).
        """
        self._snippet = {'qpos': qpos, 'qvel': qvel}
        
    def get_trajectory(self, traj_idx: int):
        del traj_idx  # Unused.
        if not hasattr(self, '_snippet'):
            raise AttributeError(
                'Trajectory not set yet. Call set_next_trajectory first.')
        return self._snippet
    
    def get_joint_names(self):
        return []
    
    def get_site_names(self):
        return []
    
class MiceTrajectoryLoader(HDF5TrajectoryLoader):
    """Loads and serves trajectories from an HDF5 file with qpos and qvel datasets."""
    
    def __init__(
        self,
        path: str,
        traj_indices: Optional[Sequence[int]] = None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        """Initializes the trajectory loader.

        Args:
            path: Path to HDF5 dataset file with reference trajectories.
            traj_indices: List of trajectory indices to use.
            random_state: Random state for reproducibility.
        """

        self._h5 = h5py.File(path, 'r')
        # Access qpos and qvel datasets
        self._qpos_dataset = self._h5['qpos']
        self._qvel_dataset = self._h5['qvel']
        self._n_traj = len(self._qpos_dataset)  # Number of trajectories is determined by the length of qpos dataset
        
        if traj_indices is None:
            self._traj_indices = np.arange(self._n_traj)
        else:
            self._traj_indices = traj_indices

        if random_state is None:
            self._random_state = np.random.RandomState(None)
        else:
            self._random_state = random_state



    def get_trajectory(
        self,
        traj_idx: Optional[int] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Returns a trajectory from the dataset.

        Args:
            traj_idx: Index of the desired trajectory. If None, a random trajectory is selected.
            start_step: Start index for the trajectory slice.
            end_step: End index for the trajectory slice.

        Returns:
            dict: Dictionary containing qpos and qvel of the trajectory.
        """
        if traj_idx is None:
            traj_idx = self._random_state.choice(self._traj_indices)

        # Fetch qpos and qvel data from the datasets
        qpos = self._qpos_dataset[traj_idx, start_step:end_step]
        qvel = self._qvel_dataset[traj_idx, start_step:end_step]

        trajectory = {
            'qpos': qpos,
            'qvel': qvel
        }

        return trajectory