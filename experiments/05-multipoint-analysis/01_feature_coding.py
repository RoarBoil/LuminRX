import os
import requests
import numpy as np
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

# Function to fetch protein sequence from UniProt
def fetch_protein_sequence(uniprotac):
    url = f"https://www.uniprot.org/uniprot/{uniprotac}.fasta"
    response = requests.get(url)
    sequence = "".join(response.text.splitlines()[1:])  # Skip the header
    return sequence

# Function to save FASTA file
def save_fasta(file_path, header, sequence):
    with open(file_path, 'w') as fasta_file:
        fasta_file.write(f">{header}\n")
        fasta_file.write(sequence)

# Function to one-hot encode a protein sequence
def one_hot_encode(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding = np.zeros((len(sequence), len(amino_acids)), dtype=int)
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            encoding[i, aa_to_index[aa]] = 1
    return encoding

# Function to apply mutation to a protein sequence
def apply_mutation(sequence, mutation):
    position = int(mutation[1:4]) - 1  # Convert 1-based index to 0-based
    new_aa = mutation[4:]  # Extract the new amino acid
    mutated_sequence = list(sequence)
    mutated_sequence[position] = new_aa  # Apply the mutation
    return ''.join(mutated_sequence)

# Function to fetch drug fingerprints and properties from PubChem
def get_drug_fingerprint(drug_name):
    try:
        compound = pcp.get_compounds(drug_name, 'name')[0]
        molecular_weight = compound.molecular_weight if compound.molecular_weight else np.nan
        fingerprint = compound.fingerprint if compound.fingerprint else np.nan
        cactvs_fingerprint = compound.cactvs_fingerprint if hasattr(compound, 'cactvs_fingerprint') else np.nan
    except (IndexError, AttributeError):
        molecular_weight, fingerprint, cactvs_fingerprint = np.nan, np.nan, np.nan
    return molecular_weight, fingerprint, cactvs_fingerprint

# Mutation dataset (uniprot_id, variant(s), drug)
mutations = [
    ("P10721", ["D816H"], "Sunitinib"),
    ("P10721", ["L576P"], "Sunitinib"),
    ("P10721", ["D816H", "L576P"], "Sunitinib")  # Two mutations together
]

# Create directories for saving FASTA files
fasta_dir = "data/"
os.makedirs(fasta_dir, exist_ok=True)

# Initialize lists for dataframe columns
data = []
onehot_before = []
onehot_after = []

# Process each mutation scenario
for mutation in mutations:
    uniprotac = mutation[0]
    variants = mutation[1]  # This is now a list of mutations
    drug = mutation[2]

    # Fetch original sequence from UniProt
    original_sequence = fetch_protein_sequence(uniprotac)

    # Apply single or multiple mutations
    mutated_sequence = original_sequence
    mutation_labels = []  # To store mutation names for file naming
    for variant in variants:
        mutated_sequence = apply_mutation(mutated_sequence, variant)
        mutation_labels.append(variant)

    mutation_str = "_".join(mutation_labels)  # Join mutations for naming

    # Save FASTA files
    save_fasta(os.path.join(fasta_dir, f"{uniprotac}.fasta"), uniprotac, original_sequence)
    save_fasta(os.path.join(fasta_dir, f"{uniprotac}_{mutation_str}.fasta"), f"{uniprotac}_{mutation_str}", mutated_sequence)

    # One-hot encode sequences
    original_encoded = one_hot_encode(original_sequence)
    mutated_encoded = one_hot_encode(mutated_sequence)

    # Get drug fingerprint information
    molecular_weight, fingerprint, cactvs_fingerprint = get_drug_fingerprint(drug)

    # Append data to the list
    data.append({
        "uniprotac": uniprotac,
        "variant": mutation_str,  # Store all mutations together
        "drug": drug,
        "original_sequence": original_sequence,
        "mutated_sequence": mutated_sequence,
        "molecular_weight": molecular_weight,
        "fingerprint": fingerprint,
        "cactvs_fingerprint": cactvs_fingerprint,
        "onehot_before": original_encoded,
        "onehot_after": mutated_encoded
    })

    # Append one-hot encodings to lists
    onehot_before.append(original_encoded)
    onehot_after.append(mutated_encoded)

# Convert to DataFrame
df = pd.DataFrame(data)

# Print DataFrame summary
print(df.head())
'''
  uniprotac      variant       drug  ...                                 cactvs_fingerprint                                      onehot_before                                       onehot_after
0    P10721        D816H  Sunitinib  ...  1110000001111011101100010000000000000000000000...  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ...  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ...      
1    P10721        L576P  Sunitinib  ...  1110000001111011101100010000000000000000000000...  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ...  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ...      
2    P10721  D816H_L576P  Sunitinib  ...  1110000001111011101100010000000000000000000000...  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ...  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ...   
'''

# Convert cactvs_fingerprint to numerical array
def process_fingerprint(fingerprint):
    return np.array([int(x) for x in fingerprint], dtype=int)

df['fingerprint_array'] = df['cactvs_fingerprint'].apply(process_fingerprint)

# Function to extract PSSM matrix from a .pssm file
def extract_pssm(pssm_file):
    with open(pssm_file, 'r') as file:
        lines = file.readlines()

    pssm_matrix = []
    reading_pssm = False

    for line in lines:
        # Start reading after the header
        if line.startswith("Last position-specific scoring matrix computed"):
            reading_pssm = True
            continue

        if reading_pssm:
            # Check if the line starts with a digit (indicating a valid PSSM row)
            line = line.strip()
            if line and line[0].isdigit():
                parts = line.split()
                scores = parts[2:22]  # Extract the PSSM matrix values (20 columns)
                pssm_matrix.append([int(score) for score in scores])
            elif not line:  # Stop at empty lines after matrix
                break

    # Ensure we return a NumPy array
    return np.array(pssm_matrix, dtype=int)

# Initialize lists for storing PSSM matrices
wild_pssms = []
mutated_pssms = []

# Assuming the dataframe `df` has been created as in previous steps
pssm_dir = "data/"

for index, row in df.iterrows():
    uniprotac = row['uniprotac']
    mutation = row['variant']
    
    # Define file paths for wild-type and mutated PSSM files
    wild_pssm_file = os.path.join(pssm_dir, f"{uniprotac}_pssm.txt")
    mutated_pssm_file = os.path.join(pssm_dir, f"{uniprotac}_{mutation}_pssm.txt")
    
    # Extract PSSM matrices for wild-type and mutated sequences
    wild_pssm = extract_pssm(wild_pssm_file)
    mutated_pssm = extract_pssm(mutated_pssm_file)
    
    wild_pssms.append(wild_pssm)
    mutated_pssms.append(mutated_pssm)

# Add PSSM matrices to DataFrame as new columns
df['wild_pssm'] = wild_pssms
df['mutated_pssm'] = mutated_pssms
print(df.head())
'''
  uniprotac      variant       drug  ...                                  fingerprint_array                                          wild_pssm                                       mutated_pssm
0    P10721        D816H  Sunitinib  ...  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, ...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...      
1    P10721        L576P  Sunitinib  ...  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, ...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...      
2    P10721  D816H_L576P  Sunitinib  ...  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, ...  [[-1, -2, -3, -4, -2, -1, -3, -3, -2, 1, 2, -2...  [[-1, -2, -3, -4, -2, -1, -3, -3, -3, 1, 2, -2...      

'''
# Save DataFrame as CSV
df.to_pickle('data/merged_evidence_fingerprint_onehot_pssm.dataset')