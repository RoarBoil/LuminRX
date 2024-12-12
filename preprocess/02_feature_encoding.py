import pandas as pd
import os
import requests
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pubchempy as pcp
from tqdm import tqdm

'''
# feature 1: fingerprint,molecular_weight,cactvs_fingerprint
drug_table = pd.read_csv('../datasets/merged_evidence.csv')
drug_table['molecular_weight'] = 0
drug_table['fingerprint'] = 0
drug_table['cactvs_fingerprint'] = 0

for i in tqdm(range(drug_table.shape[0])):
    drugname = drug_table['drug'][i]
    try:
        compound = pcp.get_compounds(drugname,'name')[0]
    except IndexError:
        continue
    try:
        smile = compound.isomeric_smiles
    except (AttributeError,IndexError):
        smile = np.nan
    try:
        molecular_weight = compound.molecular_weight
    except AttributeError:
        molecular_weight = np.nan   
    try:
        molecular_formula = compound.molecular_formula
    except AttributeError:
        molecular_formula = np.nan
    try: 
        atom = compound.atoms
    except AttributeError:
        atom = np.nan
    try:
        fingerprint = compound.fingerprint
    except AttributeError:
        fingerprint = np.nan
    try:
        cactvs_fingerprint = compound.cactvs_fingerprint
    except AttributeError:
        cactvs_fingerprint = np.nan
    drug_table.loc[i, 'molecular_weight'] = molecular_weight
    drug_table.loc[i, 'fingerprint'] = fingerprint
    drug_table.loc[i, 'cactvs_fingerprint'] = cactvs_fingerprint
    print(drug_table)
    
print(drug_table)
drug_table.to_csv('../datasets/middlefile/merged_evidence_fingerprint.csv', index=None)
'''
'''
        gene uniprotac variant               drug        label    source       patho molecular_weight                                        fingerprint                                 cactvs_fingerprint
0    SLCO1B3    F5H094   S112P  mycophenolic acid  sensitivity  pharmgkb      Benign            320.3  00000371E0783800000000000000000000000000000120...  1110000001111000001110000000000000000000000000...
1    SLCO1B3    F5H094   S112P          sunitinib  sensitivity  pharmgkb      Benign            398.5  00000371E07BB100000000000000000000000000000162...  1110000001111011101100010000000000000000000000...
2       TBXT    O15178   G177D        flunisolide  sensitivity  pharmgkb      Benign            434.5  00000371E07839000000000000000000000000000001A2...  1110000001111000001110010000000000000000000000...
3    SLC22A1    O15245   M408V          metformin  sensitivity  pharmgkb      Benign           129.16  00000371C0638000000000000000000000000000000000...  1100000001100011100000000000000000000000000000...
4      ABCC4    O15439   G187W        latanoprost  sensitivity  pharmgkb      Benign            432.6  00000371F0783800000000000000000000000000000180...  1111000001111000001110000000000000000000000000...
..       ...       ...     ...                ...          ...       ...         ...              ...                                                ...                                                ...
565  SLCO1B1    Q9Y6L6   V174A       methotrexate  sensitivity  pharmgkb  Pathogenic            454.4  00000371E07BF800000000000000000000000000000000...  1110000001111011111110000000000000000000000000...
566  SLCO1B1    Q9Y6L6   V174A        pravastatin  sensitivity  pharmgkb  Pathogenic            424.5  00000371F0783800000000000000000000000000000000...  1111000001111000001110000000000000000000000000...
567  SLCO1B1    Q9Y6L6   V174A        repaglinide  sensitivity  pharmgkb  Pathogenic            452.6  00000371F07B3800000000000000000000000000000000...  1111000001111011001110000000000000000000000000...
568  SLCO1B1    Q9Y6L6   V174A      rosiglitazone  sensitivity  pharmgkb  Pathogenic            357.4  00000371E07B3000400000000000000000000000000160...  1110000001111011001100000000000001000000000000...
569  SLCO1B1    Q9Y6L6   V174A        simvastatin  sensitivity  pharmgkb  Pathogenic            418.6  00000371F0783800000000000000000000000000000000...  1111000001111000001110000000000000000000000000...

[570 rows x 10 columns]
'''

# feature 2: onehot
df = pd.read_csv('../datasets/middlefile/merged_evidence_fingerprint.csv')
fasta_dir = '../datasets/middlefile/fasta/'

# Function to fetch the sequence from UniProt API
def fetch_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.splitlines()
        return ''.join(lines[1:])  # Skip the first line (header)
    else:
        raise ValueError(f"Failed to fetch sequence for UniProt ID {uniprot_id}")

# Function to apply mutation to a sequence
def apply_mutation(sequence, mutation):
    original_aa = mutation[0]  # First character (original amino acid)
    position = int(mutation[1:-1]) - 1  # Middle part (position), 1-based to 0-based
    new_aa = mutation[-1]  # Last character (new amino acid)
    
    if sequence[position] != original_aa:
        raise ValueError(f"Original amino acid mismatch: expected {original_aa}, found {sequence[position]} at position {position + 1}")
    
    return sequence[:position] + new_aa + sequence[position + 1:]

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

# Process the dataframe to fetch sequences, mutate, save FASTA, and one-hot encode
fasta_dir = "../datasets/middlefile/fasta/"
os.makedirs(fasta_dir, exist_ok=True)

onehot_before = []
onehot_after = []

for index, row in df.iterrows():
    uniprotac = row['uniprotac']
    mutation = row['variant']
    
    # Fetch original sequence from UniProt API
    original_sequence = fetch_sequence(uniprotac)
    
    # Apply mutation to create the mutated sequence
    mutated_sequence = apply_mutation(original_sequence, mutation)
    
    # Save original and mutated sequences to FASTA files
    save_fasta(os.path.join(fasta_dir, f"{uniprotac}.fasta"), uniprotac, original_sequence)
    save_fasta(os.path.join(fasta_dir, f"{uniprotac}_{mutation}.fasta"), f"{uniprotac}_{mutation}", mutated_sequence)
    
    # One-hot encode the original and mutated sequences
    onehot_before.append(one_hot_encode(original_sequence))
    onehot_after.append(one_hot_encode(mutated_sequence))

# Add new columns to the dataframe
df['onehot_before'] = onehot_before
df['onehot_after'] = onehot_after
print(df)
df.to_pickle('../datasets/middlefile/merged_evidence_fingerprint_onehot.dataset')

'''
        gene uniprotac variant  ...                                 cactvs_fingerprint                                      onehot_before                                       onehot_after
0    SLCO1B3    F5H094   S112P  ...  1110000001111000001110000000000000000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
1    SLCO1B3    F5H094   S112P  ...  1110000001111011101100010000000000000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
2       TBXT    O15178   G177D  ...  1110000001111000001110010000000000000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
3    SLC22A1    O15245   M408V  ...  1100000001100011100000000000000000000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
4      ABCC4    O15439   G187W  ...  1111000001111000001110000000000000000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
..       ...       ...     ...  ...                                                ...                                                ...                                                ...
565  SLCO1B1    Q9Y6L6   V174A  ...  1110000001111011111110000000000000000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
566  SLCO1B1    Q9Y6L6   V174A  ...  1111000001111000001110000000000000000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
567  SLCO1B1    Q9Y6L6   V174A  ...  1111000001111011001110000000000000000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
568  SLCO1B1    Q9Y6L6   V174A  ...  1110000001111011001100000000000001000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...
569  SLCO1B1    Q9Y6L6   V174A  ...  1111000001111000001110000000000000000000000000...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,...

[570 rows x 12 columns]
'''

