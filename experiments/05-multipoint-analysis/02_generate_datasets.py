import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load data
df = pd.read_pickle('data/merged_evidence_fingerprint_onehot_pssm.dataset')
df = df[['uniprotac','variant','drug','molecular_weight','onehot_before','onehot_after','fingerprint_array','wild_pssm','mutated_pssm']]
print(df.head())
'''
  uniprotac      variant       drug  ...                                  fingerprint_array                                          wild_pssm                                       mutated_pssm
0    P10721        D816H  Sunitinib  ...  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, ...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...      
1    P10721        L576P  Sunitinib  ...  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, ...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...      
2    P10721  D816H_L576P  Sunitinib  ...  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, ...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...  [[-1, -2, -3, -4, -2, -1, -3, -3, -3, 1, 2, -2...      
'''

# Padding or slicing onehot and pssm features to length 1024
def pad_or_slice_features(feature, target_length=1024, mutation_index=None):
    sequence_length = feature.shape[0]  # Get the length of the sequence
    feature_dim = feature.shape[1]     # Get the feature dimension (e.g., 20 for onehot, 20 for pssm)
    
    if sequence_length < target_length:
        padding_length = target_length - sequence_length
        return np.pad(
            feature,
            pad_width=((0, padding_length), (0, 0)),  # Pad sequence dimension only
            mode='constant',
            constant_values=0
        )
    elif sequence_length > target_length:
        if mutation_index is None:
            mutation_index = sequence_length // 2  # Default to center slice if mutation_index is not provided
        start = max(0, mutation_index - target_length // 2)
        end = start + target_length
        if end > sequence_length:
            end = sequence_length
            start = end - target_length
        return feature[start:end, :]
    else:
        return feature

# Apply padding or slicing to onehot and pssm features
target_length = 1024
padded_onehot_before = []
padded_onehot_after = []
padded_wild_pssm = []
padded_mutated_pssm = []

for index, row in df.iterrows():
    if(row['variant'] == 'D816H_L576P'):
        mutation_index = 816
    else:
        mutation_index = int(row['variant'][1:-1]) - 1  # Extract mutation position
    padded_onehot_before.append(pad_or_slice_features(np.array(row['onehot_before']), target_length, mutation_index))
    padded_onehot_after.append(pad_or_slice_features(np.array(row['onehot_after']), target_length, mutation_index))
    padded_wild_pssm.append(pad_or_slice_features(np.array(row['wild_pssm']), target_length, mutation_index))
    padded_mutated_pssm.append(pad_or_slice_features(np.array(row['mutated_pssm']), target_length, mutation_index))

# Add padded features back to the dataframe
df['padded_onehot_before'] = padded_onehot_before
df['padded_onehot_after'] = padded_onehot_after
df['padded_wild_pssm'] = padded_wild_pssm
df['padded_mutated_pssm'] = padded_mutated_pssm

# Prepare fingerprint features as NumPy arrays
def prepare_fingerprint_dataset(df):
    return np.stack(df['fingerprint_array'].to_numpy())  # Shape: (num_samples, fingerprint_dim)

# Prepare TensorFlow-compatible datasets
def prepare_numpy_dataset(df):
    onehot_before = np.stack(df['padded_onehot_before'].to_numpy())  # Shape: (num_samples, 1024, 20)
    onehot_after = np.stack(df['padded_onehot_after'].to_numpy())    # Shape: (num_samples, 1024, 20)
    wild_pssm = np.stack(df['padded_wild_pssm'].to_numpy())          # Shape: (num_samples, 1024, 20)
    mutated_pssm = np.stack(df['padded_mutated_pssm'].to_numpy())    # Shape: (num_samples, 1024, 20)
    combined_features = np.concatenate([onehot_before, onehot_after, wild_pssm, mutated_pssm], axis=-1)
    return combined_features

# Extract fingerprints for testing sets
test_fingerprints = prepare_fingerprint_dataset(df)
# Prepare training and testing datasets
test_inputs = prepare_numpy_dataset(df)

# Save datasets
np.save('data/test_inputs.npy', test_inputs)
np.save('data/test_fingerprints.npy', test_fingerprints)