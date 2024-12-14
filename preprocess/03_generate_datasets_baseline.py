# generate datasets
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

df = pd.read_pickle('../datasets/middlefile/merged_evidence_fingerprint_onehot_pssm.dataset')
df = df[['gene','uniprotac','variant','drug','label','patho','source','molecular_weight','onehot_before','onehot_after','fingerprint_array','wild_pssm','mutated_pssm']]
#print(df.head())
'''
      gene uniprotac variant               drug        label   patho  \
0  SLCO1B3    F5H094   S112P  mycophenolic acid  sensitivity  Benign
1  SLCO1B3    F5H094   S112P          sunitinib  sensitivity  Benign
2     TBXT    O15178   G177D        flunisolide  sensitivity  Benign
3  SLC22A1    O15245   M408V          metformin  sensitivity  Benign
4    ABCC4    O15439   G187W        latanoprost  sensitivity  Benign

     source  molecular_weight  \
0  pharmgkb            320.30
1  pharmgkb            398.50
2  pharmgkb            434.50
3  pharmgkb            129.16
4  pharmgkb            432.60

                                       onehot_before  \
0  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
1  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
2  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
3  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
4  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...

                                        onehot_after  \
0  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
1  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
2  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
3  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
4  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...

                                   fingerprint_array  \
0  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...
1  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, ...
2  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...
3  [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, ...
4  [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...

                                           wild_pssm  \
0  [[-1, -2, -2, -3, -2, -1, -2, -3, -2, 1, 2, -2...
1  [[-1, -2, -2, -3, -2, -1, -2, -3, -2, 1, 2, -2...
2  [[-2, -2, -3, -4, -2, -2, -3, -3, -2, 1, 3, -2...
3  [[-2, -2, -3, -4, -2, -1, -3, -4, -2, 0, 1, -2...
4  [[-2, -2, -3, -4, -2, -2, -3, -4, -3, 2, 4, -3...

                                        mutated_pssm
0  [[-1, -2, -2, -3, -2, -1, -2, -3, -2, 1, 2, -2...
1  [[-1, -2, -2, -3, -2, -1, -2, -3, -2, 1, 2, -2...
2  [[-2, -2, -3, -4, -2, -2, -3, -3, -2, 1, 3, -2...
3  [[-2, -2, -3, -4, -2, -1, -3, -4, -2, 0, 1, -2...
4  [[-2, -2, -3, -4, -2, -2, -3, -4, -3, 2, 4, -3...
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Padding or slicing onehot and pssm features to length 1024
def pad_or_slice_features(feature, target_length=1024, mutation_index=None):
    """
    Pad or slice a feature (either onehot or pssm) to the target length.

    Args:
        feature (np.ndarray): Input matrix (e.g., onehot or pssm), shape (sequence_length, feature_dim).
        target_length (int): Desired length for the sequence dimension.
        mutation_index (int): Mutation position index (0-based). If None, center slicing is applied for long sequences.

    Returns:
        np.ndarray: Padded or sliced matrix with shape (target_length, feature_dim).
    """
    sequence_length = feature.shape[0]  # Get the length of the sequence
    feature_dim = feature.shape[1]     # Get the feature dimension (e.g., 20 for onehot, 20 for pssm)
    
    if sequence_length < target_length:
        # Padding: Add zeros to reach the target length
        padding_length = target_length - sequence_length
        return np.pad(
            feature,
            pad_width=((0, padding_length), (0, 0)),  # Pad sequence dimension only
            mode='constant',
            constant_values=0
        )
    elif sequence_length > target_length:
        # Slicing: Extract a window of target_length centered on the mutation
        if mutation_index is None:
            mutation_index = sequence_length // 2  # Default to center slice if mutation_index is not provided
        
        # Compute start and end indices for slicing
        start = max(0, mutation_index - target_length // 2)
        end = start + target_length
        
        # Adjust start and end if they exceed sequence bounds
        if end > sequence_length:
            end = sequence_length
            start = end - target_length
        
        return feature[start:end, :]
    else:
        # Sequence length matches target_length; return unchanged
        return feature

# Apply padding or slicing to onehot and pssm features
target_length = 1024
padded_onehot_before = []
padded_onehot_after = []
padded_wild_pssm = []
padded_mutated_pssm = []

for index, row in df.iterrows():
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

# One-hot encode the labels
# Encoding the 'label' column
label_map = {'sensitivity': 0, 'resistance': 1}
df['label_encoded'] = df['label'].map(label_map)

# Encoding the 'patho' column
patho_map = {'Benign': 0, 'Likely benign': 0, 'Pathogenic': 1, 'Likely pathogenic': 1}
df['patho_encoded'] = df['patho'].map(patho_map)

# One-hot encoding the labels for contrastive learning
df['label_onehot'] = df['label_encoded'].apply(lambda x: to_categorical(x, num_classes=2))
df['patho_onehot'] = df['patho_encoded'].apply(lambda x: to_categorical(x, num_classes=2))

# Shuffle and split dataset into training and testing sets
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Prepare TensorFlow-compatible datasets
def prepare_numpy_dataset(df):
    # Convert padded features into numpy arrays
    onehot_before = np.stack(df['padded_onehot_before'].to_numpy())  # Shape: (num_samples, 1024, 20)
    onehot_after = np.stack(df['padded_onehot_after'].to_numpy())    # Shape: (num_samples, 1024, 20)
    wild_pssm = np.stack(df['padded_wild_pssm'].to_numpy())          # Shape: (num_samples, 1024, 20)
    mutated_pssm = np.stack(df['padded_mutated_pssm'].to_numpy())    # Shape: (num_samples, 1024, 20)
    
    # Concatenate features along the last dimension (amino acid level)
    # New shape: (num_samples, 1024, 80) where 80 = 20 (onehot_before) + 20 (onehot_after) + 20 (wild_pssm) + 20 (mutated_pssm)
    combined_features = np.concatenate([onehot_before, onehot_after, wild_pssm, mutated_pssm], axis=-1)
    
    # Labels
    labels_label = np.stack(df['label_onehot'].to_numpy())  # One-hot encoded label (e.g., sensitivity/resistance)
    labels_patho = np.stack(df['patho_onehot'].to_numpy())  # One-hot encoded patho (e.g., benign/pathogenic)

    return combined_features, labels_label, labels_patho

# Prepare training and testing datasets
train_inputs, train_labels_label, train_labels_patho = prepare_numpy_dataset(train_df)
test_inputs, test_labels_label, test_labels_patho = prepare_numpy_dataset(test_df)

# Save datasets
np.save('../datasets/train_data/train_inputs.npy', train_inputs)
print(train_inputs.shape) # (513, 1024, 80)
np.save('../datasets/train_data/train_labels_label.npy', train_labels_label)
print(train_labels_label.shape) # (513, 2)
np.save('../datasets/train_data/train_labels_patho.npy', train_labels_patho)
np.save('../datasets/train_data/test_inputs.npy', test_inputs)
print(test_inputs.shape) # (57, 1024, 80)
np.save('../datasets/train_data/test_labels_label.npy', test_labels_label)
print(test_labels_label.shape) # (57, 2)
np.save('../datasets/train_data/test_labels_patho.npy', test_labels_patho)

# Save DataFrame to CSV with selected columns
columns_to_save = ['uniprotac', 'variant', 'drug', 'label', 'patho', 'source']

# Save training and testing DataFrames
train_df[columns_to_save].to_csv('../datasets/trainset.csv', index=False)
test_df[columns_to_save].to_csv('../datasets/testset.csv', index=False)

print("Datasets prepared and saved for TensorFlow.")
