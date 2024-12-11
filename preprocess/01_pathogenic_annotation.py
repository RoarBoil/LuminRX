import pandas as pd
import os
import requests

def format_mutation(mutation):
    amino_acid_map = {
        'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
        'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
        'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
        'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
    }
    wild_type = amino_acid_map[mutation[0]]
    mutated_type = amino_acid_map[mutation[-1]]
    position = mutation[1:-1]
    return f"p.{wild_type}{position}{mutated_type}"

df = pd.read_csv('../datasets/merged_evidence_patho.csv')
print(df.head())
for i in range(len(df)):
    uniprotac = df['uniprotac'][i]
    variant = df['variant'][i]

    # Define the API URL
    url = "https://www.ebi.ac.uk/proteins/api/variation/" + uniprotac + "?format=json"

    # Query API and parse results
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()  # Convert to JSON format
        features = data.get("features", [])
        target_mutation = variant
        formatted_loc = format_mutation(target_mutation)
        found = False

        # Iterate over features to find the matching mutation
        for feature in features:
            if feature.get("type") == "VARIANT":
                locations = feature.get("locations", [])
                for location in locations:
                    if location.get("loc") == formatted_loc:
                        print(f"Match found for mutation: {target_mutation}")
                        clinical_significances = feature.get("clinicalSignificances", [])
                        if clinical_significances:
                            for significance in clinical_significances:
                                type_value = significance.get("type", "Unknown")
                                df.loc[i, 'patho'] = type_value
                                print(f"Clinical Significance Type: {type_value}")
                                #print(df.head())
                                df.to_csv('../datasets/merged_evidence_patho_annotation.csv',index=False)
                        else:
                            print(f"No clinical significance found for {target_mutation}")
                            df.loc[i, 'patho'] = 'none'
                        found = True
                        break
            if found:
                break

        if not found:
            print(f"The mutation {target_mutation} was not found in the API response.")
    else:
        print(f"Failed to fetch data from API. Status code: {response.status_code}")

print(df.head())
'''
      gene uniprotac variant               drug                                              smile  label    source   patho
0  SLCO1B3    F5H094   M233I  mycophenolic acid     CC1=C2COC(=O)C2=C(C(=C1OC)C/C=C(\C)/CCC(=O)O)O      1  pharmgkb    none
1  SLCO1B3    F5H094   S112P  mycophenolic acid     CC1=C2COC(=O)C2=C(C(=C1OC)C/C=C(\C)/CCC(=O)O)O      1  pharmgkb  Benign
2  SLCO1B3    F5H094   S112P          sunitinib  CCN(CC)CCNC(=O)C1=C(NC(=C1C)/C=C\2/C3=C(C=CC(=...      1  pharmgkb  Benign
3     TBXT    O15178   G177D        flunisolide  C[C@]12C[C@@H]([C@H]3[C@H]([C@@H]1C[C@@H]4[C@]...      1  pharmgkb  Benign
4  SLC22A1    O15245   M408V          metformin                                  CN(C)C(=N)N=C(N)N      1  pharmgkb  Benign
'''
df.to_csv('../datasets/merged_evidence_patho_annotation.csv',index=False)