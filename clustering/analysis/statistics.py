import os
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

class H5Statistics():
    def __init__(self, res_dir, participants_tsv):
        """
        Initialize the H5Statistics class.

        Parameters:
        res_dir (str): The directory containing the HDF5 files.
        participants_tsv (str): The path to the participants TSV file with demographic information.
        """
        self.res_dir = res_dir
        self.participants_tsv = participants_tsv
        self.h5_files = self._load_tract_h5s()
        print(len(self.h5_files.keys()))
        self.demographics = self._load_demographics()

    def _load_tract_h5s(self):
        """
        Load all HDF5 files in the specified directory using glob.

        Returns:
        dict: A dictionary where keys are filenames and values are HDF5 file handles.
        """
        h5_files = {}
        h5_paths = glob.glob(os.path.join(self.res_dir, '*.h5'))
        for file_path in h5_paths:
            filename = os.path.basename(file_path)
            h5_files[filename] = h5py.File(file_path, 'r')
        return h5_files

    def _load_demographics(self):
        """
        Load the participants TSV file with demographic information.

        Returns:
        pandas.DataFrame: A DataFrame containing the demographic information.
        """
        return pd.read_csv(self.participants_tsv, sep='\t')

    def extract_statistics(self):
        """
        Extract the mean statistics from the HDF5 files and combine with demographic information.

        Returns:
        pandas.DataFrame: A DataFrame containing the combined data.
        """
        data = []
        for filename, h5_file in self.h5_files.items():
            mean_fa = h5_file.attrs.get('mean_fa')
            mean_ad = h5_file.attrs.get('mean_ad')
            mean_rd = h5_file.attrs.get('mean_rd')
            mean_md = h5_file.attrs.get('mean_md')
            side_id = h5_file.attrs.get('side_id')
            participant_id = h5_file.attrs.get('sub_id')
            group = self.demographics.loc[self.demographics['participant_id'] == participant_id, 'Group'].values

            if len(group) == 0:
                print(f"No group information found for participant ID: {participant_id}")
                continue


            group = group[0]
            data.append({
                'participant_id': participant_id,
                'group': group,
                'side_id': side_id,
                'mean_fa': mean_fa,
                'mean_ad': mean_ad,
                'mean_rd': mean_rd,
                'mean_md': mean_md
            })

        df = pd.DataFrame(data)
        # Handle missing values
        df.dropna(inplace=True)
        print("Extracted DataFrame:")
        print(df.head())
        return df

    def plot_statistics(self):
        """
        Plot the statistics comparing mean_fa, mean_ad, mean_rd, mean_md for left and right tracts,
        each subdivided by patient and control groups. Also, perform t-tests to determine the significance.
        """
        df = self.extract_statistics()
        print("DataFrame columns:", df.columns)

        required_columns = ['side_id', 'group', 'mean_fa', 'mean_ad', 'mean_rd', 'mean_md']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        metrics = ['mean_fa', 'mean_ad', 'mean_rd', 'mean_md']
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='side_id', y=metric, hue='group', data=df)
            plt.title(f'Comparison of {metric} for left and right tracts (patients vs controls)')
            plt.xlabel('Side ID')
            plt.ylabel(metric.upper())
            plt.legend(title='Group')

            # Perform Welch's t-tests for left and right sides
            for side in ['lh', 'rh']:
                patients = df[(df['side_id'] == side) & (df['group'] == 'patient')][metric]
                controls = df[(df['side_id'] == side) & (df['group'] == 'control')][metric]

                print(f"Side {side}, metric {metric} - Patients: {len(patients)}, Controls: {len(controls)}")

                # Check for sufficient data before performing the t-test
                if len(patients) > 1 and len(controls) > 1:
                    t_stat, p_value = ttest_ind(patients, controls, equal_var=False)
                else:
                    t_stat, p_value = float('nan'), float('nan')
                    print(f"Insufficient data for t-test for {metric} on side {side}.")

                print(f"T-test for {metric} on side {side}: t={t_stat}, p={p_value}")

                # Annotate the plot with the p-value
                y_max = df[metric].max()
                x_pos = -0.2 if side == 'lh' else 0.2
                plt.text(x_pos, y_max, f'p={p_value:.3e}', horizontalalignment='center', color='red', weight='bold')

            plt.show()
