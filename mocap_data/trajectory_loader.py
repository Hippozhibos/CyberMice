import pandas as pd

class CSVTrajectoryLoader:
    def __init__(self, file_path, proto_modifier=None):
        self.file_path = file_path
        self.proto_modifier = proto_modifier
        self.data = self._load_data()

    def _load_data(self):
        # Assuming the CSV file has a specific structure
        data = pd.read_csv(self.file_path)
        if self.proto_modifier:
            data = self.proto_modifier(data)
        return data

    def get_trajectory(self, trajectory_id):
        # Retrieve a specific trajectory by id
        trajectory = self.data[self.data['trajectory_id'] == trajectory_id]
        if trajectory.empty:
            raise ValueError(f"Trajectory ID {trajectory_id} not found")
        return trajectory


    def _load_reference_data(self, ref_path, proto_modifier, dataset, file_format="hdf5"):
        if file_format == "hdf5":
            self._loader = HDF5TrajectoryLoader(ref_path, proto_modifier=proto_modifier)
        elif file_format == "csv":
            self._loader = CSVTrajectoryLoader(ref_path, proto_modifier=proto_modifier)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        self.reference_data = self._loader.data
        self.reference_trajectories = dataset.trajectories

