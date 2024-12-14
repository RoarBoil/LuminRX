import os
import subprocess

# BLAST binary and database paths
bin_path = '/data3/liuz/blast/bin/psiblast'
db_path = '/data3/liuz/blast/db/swissprot'
output_dir = "../datasets/middlefile/blast_results/" 
fasta_dir = "../datasets/middlefile/fasta/"  

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all FASTA files
fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith(".fasta")]

# Run BLAST for each FASTA file
for fasta_file in fasta_files:
    fasta_path = os.path.join(fasta_dir, fasta_file)
    
    # Define output paths
    base_name = os.path.splitext(fasta_file)[0]
    blast_path = os.path.join(output_dir, f"{base_name}_blast.txt")
    pssm_path = os.path.join(output_dir, f"{base_name}_pssm.txt")
    
    # Construct the BLAST command
    cmd = (
        f"{bin_path} -db {db_path} -query {fasta_path} "
        f"-out {blast_path} -inclusion_ethresh 0.001 "
        f"-out_ascii_pssm {pssm_path} -num_iterations 3 -num_threads 8"
    )
    
    # Print the command for debugging
    print(f"Running BLAST for {fasta_file}...")
    print(f"Command: {cmd}")
    
    # Execute the command
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"BLAST completed for {fasta_file}. Results saved to {blast_path} and {pssm_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running BLAST for {fasta_file}: {e}")

print("All BLAST tasks completed.")