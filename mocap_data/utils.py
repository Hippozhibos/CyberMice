import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle as pkl

def load_syllable(result_path):
    '''
    Load the syllable data from .npy or .h5 file

    Parameters
    ----------
    result_path: str
        Path to the .npy or .h5 file containing the syllable data.
        
    Returns
    -------
    syllable: ndarray of shape (num_frames,)
    '''
    whether_h5 = result_path.endswith('.h5')
    if whether_h5:
        h5_data = h5py.File(result_path, 'r')
        # syllable = h5_data['data']['syllable'][:] 
        syllable = h5_data['data_with_ephys']['syllable'][:] # former version
    else:
        syllable = np.load(result_path)
    return syllable

def rle(syllable):
    '''
    Compute the run length encoding of a discrete state sequence

    E.g. the state sequence [0, 0, 1, 1, 1, 2, 3, 3]
         would be encoded as ([0, 1, 2, 3], [2, 3, 1, 2])

    [Copied from pyhsmm.util.general.rle]

    Parameters
    ----------
    syllable : ndarray of shape (num_frames,)
        discrete state sequence

    Returns
    -------
    ids : ndarray 
        integer identities of the states

    durations : ndarray with the same shape as ids
        length of time in corresponding state
    '''
    pos, = np.where(np.diff(syllable) != 0)
    pos = np.concatenate(([0],pos+1,[len(syllable)]))
    return syllable[pos[:-1]], np.diff(pos)

def get_duration_list(syllable):
    '''
    Compute the duration of each state in run length encoding of states

    Parameters
    ----------
    syllable : ndarray of shape (num_frames,)

    Returns
    -------
    inferred_state_list: list of int
        indicates the state identity for continuous behaviors
    
    inferred_durations: list of int
        indicates the duration of each inferred state instance
    
    inf_durs_stacked: list of int
        rearrange the lists of durations to be a nested list where the nth inner list is a list of durations for state n
    '''
    inferred_state_list, inferred_durations = rle(syllable)
    num_states = np.unique(syllable).shape[0]    # number of discrete states

    inf_durs_stacked = []
    for s in range(num_states):
        inf_durs_stacked.append(inferred_durations[inferred_state_list == s])
    
    return inferred_state_list, inferred_durations, inf_durs_stacked

def get_sig_list(syllable, threshold_ratio=0.005):
    '''
    Compute the frequency of each state in run length encoding of states
    
    Parameters
    ----------
    syllable : ndarray of shape (num_frames,)
        discrete state sequence

    threshold_ratio : float, optional
        set for threshold of syllable frequency, default 0.005

    Returns 
    -------
    sig_list : list of int
        indicates syllables appear more frequent than threshold_ratio*total_syllables
    
    frequency_dict : dict
        key: each state identity, value: frequency of state identity
    '''
    # show state frequency in each state
    sig_list = []
    frequency_dict={}
    inferred_state_list,_ = rle(syllable)
    key = np.unique(inferred_state_list)
    threshold = inferred_state_list.shape[0]*threshold_ratio
    for k in key:
        v = inferred_state_list[inferred_state_list == k].size
        frequency_dict[k] = v
        if v > threshold:
            sig_list.append(k)
    return sig_list, frequency_dict

def get_pose(pose_path, concatenate=False):
    '''
    get pose from .npy file

    Parameters:
    ----------
    pose_path: str, path to the pose file
    concatenate: bool, whether to concatenate all the pose data in the directory

    Returns:
    -------    
    coordinates: dict, contains the pose data in the format of {'data': coordinates_arr}
        coordinate_arr: ndarray of shape (timepoints, keypoints, dimensions)
    '''
    if concatenate:
        data_list = os.listdir(pose_path) # all data in data_path should be .npy file
        coors_to_concatenate = []
        for data in data_list:
            try:
                assert data.endswith('.npy')
            except AssertionError:
                print('file format is not npy')

            pose = np.load(pose_path+'/'+data)
            coordinates_arr = pose.transpose(0, 2, 1)
            coors_to_concatenate.append(coordinates_arr)
        concatenated_coors = np.concatenate(coors_to_concatenate, axis=0)
        coordinates = {'data': concatenated_coors}
        return coordinates, data_list
    else:
        try:
            assert pose_path.endswith('.npy')
        except AssertionError:
            print('file format is not npy')

        pose = np.load(pose_path)
        coordinates_arr = pose.transpose(0, 2, 1)
        coordinates = {'data': coordinates_arr}
        return coordinates

def select_kp(all_pose, all_syllable, select_syllable, syllable_name, min_len=10):
    '''
        Select the keypoints of the specified syllable from the pose data

    Parameters
    ----------
    all_pose: dict, contains the pose data in the format of {'data': coordinates_arr}
        coordinates_arr: ndarray of shape (timepoints, keypoints, dimensions)
    
    all_syllable: ndarray of shape (num_frames,)

    select_syllable: int, the index of the selected syllable
    
    syllable_name: str, the name of the selected syllable
    
    min_len: int, the minimum duration length of the selected syllable

    Returns
    -------
    select_kp_list: list of ndarray
        each ndarray contains consecutive keypoints of the selected syllable
    '''
    all_coor = all_pose['data']
    syllable_change, syllable_len = rle(all_syllable)
    total_pos = np.where(np.diff(all_syllable) != 0)[0] + 1
    pos_add0 = np.concatenate(([0], total_pos))

    select_kp_list = []
    select_pos = pos_add0[syllable_change == select_syllable]
    select_duration = syllable_len[syllable_change == select_syllable]
    for pos,duration in zip(select_pos, select_duration):
        if duration > min_len:
            select_kp_list.append(all_coor[pos:pos+duration, :, :])
    with open(f'{syllable_name}_kp_list.pkl', 'wb') as f:
        pkl.dump(select_kp_list, f)
    return select_kp_list
    