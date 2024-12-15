import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load data
df = pd.read_pickle('../datasets/middlefile/merged_evidence_fingerprint_onehot_pssm.dataset')
df = df[['gene','uniprotac','variant','drug','label','patho','source','molecular_weight','onehot_before','onehot_after','fingerprint_array','wild_pssm','mutated_pssm']]

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
label_map = {'sensitivity': 0, 'resistance': 1}
df['label_encoded'] = df['label'].map(label_map)

patho_map = {'Benign': 0, 'Likely benign': 0, 'Pathogenic': 1, 'Likely pathogenic': 1}
df['patho_encoded'] = df['patho'].map(patho_map)

df['label_onehot'] = df['label_encoded'].apply(lambda x: to_categorical(x, num_classes=2))
df['patho_onehot'] = df['patho_encoded'].apply(lambda x: to_categorical(x, num_classes=2))

# Create a unique mutation identifier
df['mutation_id'] = df['uniprotac'] + '_' + df['variant']

# Unique mutation list for splitting
unique_mutations = df['mutation_id'].unique()

# Split mutations into training and testing
train_mutations, test_mutations = train_test_split(
    unique_mutations, test_size=0.1, random_state=42
)

# Split data based on mutation groups
train_df = df[df['mutation_id'].isin(train_mutations)].reset_index(drop=True)
test_df = df[df['mutation_id'].isin(test_mutations)].reset_index(drop=True)

# Drop the temporary mutation_id column
train_df.drop(columns=['mutation_id'], inplace=True)
test_df.drop(columns=['mutation_id'], inplace=True)

# Prepare TensorFlow-compatible datasets
def prepare_numpy_dataset(df):
    onehot_before = np.stack(df['padded_onehot_before'].to_numpy())  # Shape: (num_samples, 1024, 20)
    onehot_after = np.stack(df['padded_onehot_after'].to_numpy())    # Shape: (num_samples, 1024, 20)
    wild_pssm = np.stack(df['padded_wild_pssm'].to_numpy())          # Shape: (num_samples, 1024, 20)
    mutated_pssm = np.stack(df['padded_mutated_pssm'].to_numpy())    # Shape: (num_samples, 1024, 20)
    
    combined_features = np.concatenate([onehot_before, onehot_after, wild_pssm, mutated_pssm], axis=-1)
    labels_label = np.stack(df['label_onehot'].to_numpy())  # One-hot encoded label
    labels_patho = np.stack(df['patho_onehot'].to_numpy())  # One-hot encoded patho

    return combined_features, labels_label, labels_patho

# Prepare training and testing datasets
train_inputs, train_labels_label, train_labels_patho = prepare_numpy_dataset(train_df)
test_inputs, test_labels_label, test_labels_patho = prepare_numpy_dataset(test_df)

# Print dataset summary
print(f"Training set: {len(train_df)} samples, {len(train_mutations)} unique mutations")
print(f"Testing set: {len(test_df)} samples, {len(test_mutations)} unique mutations")

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
train_df[columns_to_save].to_csv('../datasets/trainset_unique.csv', index=False)
test_df[columns_to_save].to_csv('../datasets/testset_unique.csv', index=False)

print("Datasets prepared and saved for TensorFlow.")