import h5py

def load_training_data(file_path: str):
    """
    Utility function to load the training data from HDF5 file.
    
    Returns:
        dict: Dictionary containing the loaded data
    """
    with h5py.File(file_path, 'r') as hf:
        data = {
            'latent_vectors': hf['latent_vectors'][:],
            'accuracy_rewards': hf['accuracy_rewards'][:],
            'format_rewards': hf['format_rewards'][:],
            'example_ids': hf['example_ids'][:],
            'metadata': dict(hf.attrs)
        }
    return data
