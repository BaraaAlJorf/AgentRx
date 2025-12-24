import json

def view_patient_data_from_jsonl(file_path):
    """
    Loads data from a JSON Lines (.jsonl) file and interactively
    displays the information for each patient record.

    Args:
        file_path (str): The path to the JSON Lines file.
    """
    patient_records = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Strip whitespace and check for empty lines
                line = line.strip()
                if line:
                    # Parse each line as a separate JSON object
                    patient_records.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: A line in '{file_path}' contains invalid JSON. Could not parse.")
        print(f"   Error details: {e}")
        return

    total_patients = len(patient_records)

    if total_patients == 0:
        print(f"🤷 No patient records found in '{file_path}'.")
        return

    print(f"✅ Successfully loaded {total_patients} patient records from '{file_path}'.\n")

    for i, record in enumerate(patient_records):
        # In a .jsonl file, there isn't a top-level patient ID key
        # We'll just display the records in order
        print("=" * 60)
        print(f"👨‍⚕️ Patient Record {i + 1}/{total_patients}")
        print("-" * 60)

        # Use json.dumps() with indentation for a "nice" print format
        print(json.dumps(record, indent=4))
        print("=" * 60 + "\n")

        # Pause and wait for user input
        if i < total_patients - 1:
            try:
                user_input = input("Press Enter for the next record, or type 'q' and Enter to quit: ")
                if user_input.lower() == 'q':
                    print("\nExiting viewer.")
                    break
            except KeyboardInterrupt:
                print("\n\nExiting viewer.")
                break

if __name__ == "__main__":
    # ⬇️ IMPORTANT: Make sure this is your .jsonl file name!
    jsonl_filename = "/scratch/baj321/MedAgent/datasets/multimodal_dataset_splits/test.jsonl"
    view_patient_data_from_jsonl(jsonl_filename)