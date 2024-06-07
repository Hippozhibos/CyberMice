"""Datasets for reference pose tasks.
"""

from dm_control.locomotion.tasks.reference_pose import cmu_subsets

# Define your custom dataset
CUSTOM_DATASETS = {
    'diving': 'D:\Mujoco\CyberMice\mocap_data\mocap_data\diving',
    # Add more datasets as needed
}

DATASETS = dict()
DATASETS.update(cmu_subsets.CMU_SUBSETS_DICT)
