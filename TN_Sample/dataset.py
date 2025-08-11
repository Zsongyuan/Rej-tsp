import json
import random

def create_mixed_dataset(negative_file, positive_file, output_file, num_samples=36665):
    """
    Combines negative and positive samples into a single JSON file for training.

    Args:
        negative_file (str): Path to the JSON file with negative samples.
        positive_file (str): Path to the JSON file with positive samples.
        output_file (str): Path to save the combined JSON file.
        num_samples (int): The number of samples to take from each file.
    """
    try:
        # Load negative samples
        with open(negative_file, 'r', encoding='utf-8') as f:
            negative_data = json.load(f)
        print(f"Successfully loaded {len(negative_data)} negative samples from {negative_file}")

        # Load positive samples
        with open(positive_file, 'r', encoding='utf-8') as f:
            positive_data = json.load(f)
        print(f"Successfully loaded {len(positive_data)} positive samples from {positive_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the JSON files are in the same directory as this script.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from a file: {e}")
        return

    # Take the first num_samples and add the 'is_negative' flag
    processed_negatives = []
    for sample in negative_data[:num_samples]:
        sample['is_negative'] = True
        processed_negatives.append(sample)

    processed_positives = []
    for sample in positive_data[:num_samples]:
        sample['is_negative'] = False
        processed_positives.append(sample)

    # Combine and shuffle the data
    combined_data = processed_negatives + processed_positives
    random.shuffle(combined_data)

    # Save the combined data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4)

    print(f"\nSuccessfully created mixed dataset with {len(combined_data)} total samples.")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    # Define your file paths
    negatives_json = './sampled_negatives_36665.json'
    positives_json = '../ScanRefer/ScanRefer_filtered_train.json'
    output_json = 'train_mixed_36665.json'

    # Run the script
    create_mixed_dataset(negatives_json, positives_json, output_json, num_samples=36665)