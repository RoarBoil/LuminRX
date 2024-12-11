import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pubchempy as pcp
from tqdm import tqdm

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
