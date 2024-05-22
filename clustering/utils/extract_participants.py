import csv
import sys

def extract_participant_ids(tsv_path):
    participant_ids = []
    with open(tsv_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            participant_ids.append(row['participant_id'])
    return participant_ids

if __name__ == "__main__":
    tsv_path = sys.argv[1]
    participant_ids = extract_participant_ids(tsv_path)
    for pid in participant_ids:
        print(pid)