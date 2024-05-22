import pandas as pd
import sys

def get_participant_info(tsv_file, participant_id):
    # Load the TSV file into a DataFrame
    df = pd.read_csv(tsv_file, sep='\t')

    # Query for the participant
    participant_data = df[df['participant_id'] == participant_id].iloc[0]

    # Extract the desired information
    group = participant_data['Group']
    fmri = participant_data['FMRI']
    pid = participant_data['ID']
    affected_hemisphere = participant_data['AffectedHemisphere']
    excluded = participant_data['Excluded']

    return group, int(fmri), int(pid), affected_hemisphere, excluded

if __name__ == "__main__":
    # The first argument is the script name, so the second is the TSV file and the third is the participant ID
    if len(sys.argv) != 3:
        print("Usage: python script.py data.tsv participant_id")
        sys.exit(1)

    # Get the TSV file path and participant ID from the command line
    tsv_file = sys.argv[1]
    participant_id = sys.argv[2]

    # Get participant info
    group, fmri, id, affected_hemisphere, excluded = get_participant_info(tsv_file, participant_id)

    # Output the information so it can be captured by a Bash script
    print(f"{group}\t{fmri}\t{id}\t{affected_hemisphere}\t{excluded}")
