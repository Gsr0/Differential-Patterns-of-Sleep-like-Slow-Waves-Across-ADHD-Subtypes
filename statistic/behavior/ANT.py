import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


class ANTDataExtractor:
    def __init__(self):
        """
        ANT Data Extractor
        Extract ANT three network indicators from E-Prime .edat3 files or text files
        """
        self.data = None
        self.results = None

    def parse_eprime_text(self, file_path):
        """
        Parse E-Prime text file

        Parameters:
        -----------
        file_path : str
            File path

        Returns:
        --------
        pd.DataFrame : Parsed dataframe
        """
        data_list = []
        current_trial = {}

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

        # Extract basic information
        header_match = re.search(r'\*\*\* Header Start \*\*\*(.*?)\*\*\* Header End \*\*\*',
                                 content, re.DOTALL)

        subject_id = None
        session_date = None
        group = None

        if header_match:
            header_content = header_match.group(1)
            subject_match = re.search(r'Subject: (\w+)', header_content)
            date_match = re.search(r'SessionDate: ([\d-]+)', header_content)
            group_match = re.search(r'Group: (\d+)', header_content)

            if subject_match:
                subject_id = subject_match.group(1)
            if date_match:
                session_date = date_match.group(1)
            if group_match:
                group = int(group_match.group(1))

        # Extract all LogFrames
        logframes = re.findall(r'\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*',
                               content, re.DOTALL)

        for frame in logframes:
            trial_data = {
                'Subject': subject_id,
                'SessionDate': session_date,
                'Group': group
            }

            # Parse each line of data
            lines = frame.strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('Level'):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    # Convert numeric values
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)

                    trial_data[key] = value

            # Only keep formal trial data (not practice trials)
            if trial_data.get('Procedure') != 'PRAC':
                data_list.append(trial_data)

        return pd.DataFrame(data_list)

    def load_edat3_file(self, file_path):
        """
        Load .edat3 file (needs to be converted to text format first)

        Parameters:
        -----------
        file_path : str
            .edat3 file path
        """
        # Note: .edat3 files usually need E-DataAid software to convert to text format
        # Here we assume it has been converted to text format
        print("Note: .edat3 files need to be converted to text format using E-DataAid first")
        print("Or provide the converted text file directly")

    def load_data(self, file_path):
        """
        Load data file

        Parameters:
        -----------
        file_path : str
            File path (supports .txt format)
        """
        if file_path.endswith('.txt'):
            self.data = self.parse_eprime_text(file_path)
        elif file_path.endswith('.edat3'):
            print("Please convert .edat3 file to text format first, or export as .txt file using E-DataAid")
            return None
        else:
            print("Unsupported file format. Please use .txt or .edat3 files")
            return None

        print(f"Successfully loaded data with {len(self.data)} trials")
        return self.data

    def clean_data(self, min_rt=200, max_rt=2000):
        """
        Clean data

        Parameters:
        -----------
        min_rt : int
            Minimum reaction time (milliseconds)
        max_rt : int
            Maximum reaction time (milliseconds)
        """
        if self.data is None:
            print("Please load data first")
            return

        original_count = len(self.data)

        # Remove trials with missing reaction times
        self.data = self.data.dropna(subset=['Target.RT'])

        # Remove trials with RT = 0 (possibly no response)
        self.data = self.data[self.data['Target.RT'] > 0]

        # Remove trials with abnormal reaction times
        self.data = self.data[
            (self.data['Target.RT'] >= min_rt) &
            (self.data['Target.RT'] <= max_rt)
            ]

        # Keep only correct trials for RT analysis (accuracy analysis needs all trials)
        self.data_correct = self.data[self.data['Target.ACC'] == 1].copy()

        cleaned_count = len(self.data)
        print(f"Data cleaning completed: {original_count} -> {cleaned_count} trials")
        print(f"Number of correct trials: {len(self.data_correct)}")

    def calculate_ant_networks(self):
        """
        Calculate effect values for ANT three networks

        Returns:
        --------
        dict : Dictionary containing effect values for three networks
        """
        if self.data is None:
            print("Please load data first")
            return None

        # Ensure data has been cleaned
        if not hasattr(self, 'data_correct'):
            self.clean_data()

        # Calculate mean RT grouped by conditions
        condition_rt = self.data_correct.groupby(['Cue', 'Flanker'])['Target.RT'].mean()

        results = {}

        try:
            # 1. Alerting Network Effect
            # No cue condition - Central cue condition
            no_cue_rt = condition_rt.get(('No_Cue', 'Congruent'), 0) + \
                        condition_rt.get(('No_Cue', 'Incongruent'), 0) + \
                        condition_rt.get(('No_Cue', 'Neutral'), 0)
            no_cue_rt = no_cue_rt / 3  # Take average

            center_cue_rt = condition_rt.get(('Center_Cue', 'Congruent'), 0) + \
                            condition_rt.get(('Center_Cue', 'Incongruent'), 0) + \
                            condition_rt.get(('Center_Cue', 'Neutral'), 0)
            center_cue_rt = center_cue_rt / 3  # Take average

            alerting_effect = no_cue_rt - center_cue_rt
            results['Alerting_Effect'] = alerting_effect

            # 2. Orienting Network Effect
            # Central cue condition - Spatial cue condition
            spatial_cue_rt = condition_rt.get(('S_Cue', 'Congruent'), 0) + \
                             condition_rt.get(('S_Cue', 'Incongruent'), 0) + \
                             condition_rt.get(('S_Cue', 'Neutral'), 0)
            spatial_cue_rt = spatial_cue_rt / 3  # Take average

            orienting_effect = center_cue_rt - spatial_cue_rt
            results['Orienting_Effect'] = orienting_effect

            # 3. Executive Network Effect
            # Incongruent condition - Congruent condition
            incongruent_rt = (condition_rt.get(('No_Cue', 'Incongruent'), 0) +
                              condition_rt.get(('Center_Cue', 'Incongruent'), 0) +
                              condition_rt.get(('S_Cue', 'Incongruent'), 0)) / 3

            congruent_rt = (condition_rt.get(('No_Cue', 'Congruent'), 0) +
                            condition_rt.get(('Center_Cue', 'Congruent'), 0) +
                            condition_rt.get(('S_Cue', 'Congruent'), 0)) / 3

            executive_effect = incongruent_rt - congruent_rt
            results['Executive_Effect'] = executive_effect

        except Exception as e:
            print(f"Error calculating network effects: {e}")
            print("Available condition combinations:")
            print(condition_rt.index.tolist())

        return results

    def calculate_accuracy_rt(self):
        """
        Calculate overall accuracy and reaction time

        Returns:
        --------
        dict : Dictionary containing accuracy and reaction time measures
        """
        if self.data is None:
            print("Please load data first")
            return None

        results = {}

        # Overall accuracy
        total_accuracy = self.data['Target.ACC'].mean()
        results['Overall_Accuracy'] = total_accuracy

        # Accuracy by condition
        cue_accuracy = self.data.groupby('Cue')['Target.ACC'].mean()
        flanker_accuracy = self.data.groupby('Flanker')['Target.ACC'].mean()

        for cue_type, acc in cue_accuracy.items():
            results[f'{cue_type}_Accuracy'] = acc

        for flanker_type, acc in flanker_accuracy.items():
            results[f'{flanker_type}_Accuracy'] = acc

        # Reaction time (correct trials only)
        if hasattr(self, 'data_correct'):
            overall_rt = self.data_correct['Target.RT'].mean()
            results['Overall_RT'] = overall_rt

            # Reaction time by condition
            cue_rt = self.data_correct.groupby('Cue')['Target.RT'].mean()
            flanker_rt = self.data_correct.groupby('Flanker')['Target.RT'].mean()

            for cue_type, rt in cue_rt.items():
                results[f'{cue_type}_RT'] = rt

            for flanker_type, rt in flanker_rt.items():
                results[f'{flanker_type}_RT'] = rt

        return results

    def extract_all_measures(self):
        """
        Extract all ANT measures

        Returns:
        --------
        dict : Dictionary containing all measures
        """
        if self.data is None:
            print("Please load data first")
            return None

        # Basic information
        subject_info = {
            'Subject': self.data['Subject'].iloc[0] if 'Subject' in self.data.columns else 'Unknown',
            'Group': self.data['Group'].iloc[0] if 'Group' in self.data.columns else 'Unknown',
            'SessionDate': self.data['SessionDate'].iloc[0] if 'SessionDate' in self.data.columns else 'Unknown'
        }

        # Network effects
        network_effects = self.calculate_ant_networks()

        # Accuracy and reaction time measures
        acc_rt_measures = self.calculate_accuracy_rt()

        # Merge all results
        all_results = {**subject_info, **network_effects, **acc_rt_measures}

        return all_results

    def process_multiple_files(self, folder_path, output_file='ant_results.csv'):
        """
        Process multiple files in batch

        Parameters:
        -----------
        folder_path : str
            Folder path containing ANT data files
        output_file : str
            Output CSV filename
        """
        all_results = []

        # Find all text files
        folder = Path(folder_path)
        txt_files = list(folder.glob('*.txt'))

        print(f"Found {len(txt_files)} text files")

        for file_path in txt_files:
            print(f"Processing file: {file_path.name}")

            try:
                # Load data
                self.load_data(str(file_path))

                # Extract measures
                results = self.extract_all_measures()

                if results:
                    all_results.append(results)
                    print(f"  Data extraction successful")
                else:
                    print(f"  Data extraction failed")

            except Exception as e:
                print(f"  Error processing file: {e}")

        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            print(f"Processed data from {len(all_results)} subjects")

            # Display result summary
            print("\nResult summary:")
            print(results_df.describe())

            return results_df
        else:
            print("Could not extract any data")
            return None


# Usage example
def main():
    """
    Main function - Usage example
    """
    # Create extractor instance
    extractor = ANTDataExtractor()

    # Method 1: Process single file
    print("=== Single file processing example ===")
    # file_path = "path/to/your/ant_data.txt"  # Replace with your file path
    # extractor.load_data(file_path)
    # results = extractor.extract_all_measures()
    # print("Extracted measures:")
    # for key, value in results.items():
    #     print(f"{key}: {value}")

    # Method 2: Batch process multiple files
    print("=== Batch processing example ===")
    folder_path = "D:\桌面\正常儿童\\0525cyx\cyxbhv"  # Replace with your data folder path
    results_df = extractor.process_multiple_files(folder_path, 'ant_network_results.csv')

    print("Please uncomment the above code and modify the file path before running")


if __name__ == "__main__":
    main()