# Import the function we want to test from data_utils.py
from data_utils import get_data_loader

def run_test():
    """
    Tests the get_data_loader function with a sample data file.
    """
    print("--- Starting DataLoader Test ---")

    # --- Configuration ---
    # !! IMPORTANT: Change this to the path of your REAL json file when ready.
    test_file_path = '/scratch/baj321/MedAgent/test_dataset.jsonl'
    batch_size = 2
    num_workers = 0  # Use 0 for simple tests to avoid multiprocessing issues

    # --- 1. Create the DataLoader ---
    try:
        data_loader = get_data_loader(
            data_path=test_file_path,
            batch_size=batch_size,
            num_workers=num_workers
        )
        print(f"✅ Successfully created DataLoader for '{test_file_path}'.")
        print(f"Total patients in dataset: {len(data_loader.dataset)}")
        print(f"Batch size: {batch_size}")
        print("-" * 30)

    except Exception as e:
        print(f"❌ Failed to create DataLoader. Error: {e}")
        return

    # --- 2. Iterate through the DataLoader ---
    print("Iterating through batches...")
    total_batches = 0
    total_patients_processed = 0

    try:
        for i, batch in enumerate(data_loader):
            total_batches += 1
            total_patients_processed += len(batch)
            
            print(f"\n--- Batch {i+1} ---")
            print(f"Type of batch: {type(batch)}")
            print(f"Number of patients in this batch: {len(batch)}")

            # Let's inspect the first patient in the batch
            if batch:
                first_patient = batch[0]
                print(f"Type of first patient: {type(first_patient)}")
                print(f"Stay ID of first patient: {first_patient.get('stay_id')}")
                print(f"Keys available for first patient: {list(first_patient.keys())}")
        
        print("\n" + "-" * 30)
        print("✅ Successfully iterated through all batches.")
        print(f"Total batches generated: {total_batches}")
        print(f"Total patients processed: {total_patients_processed}")

    except Exception as e:
        print(f"❌ An error occurred while iterating through the DataLoader: {e}")

if __name__ == '__main__':
    run_test()