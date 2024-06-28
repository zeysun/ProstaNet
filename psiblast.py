import os
import subprocess

# Generate position-specific scoring matrix by using PSI-BLAST
fasta_path = '/home/til60/Desktop/Protein_stability/CD4/fasta' #protein fasta files
output_path = '/home/til60/Desktop/Blast/blast_J3'
script_path = '/home/til60/Desktop/Blast/bin/psiblast'
db_path = '/home/til60/Desktop/Blast/UniRef/uniref90.fasta' #Uniref90 database file

# PSI-Blast cmd
def run_psiblast(file_path, name):
    psiblast_args = [
        script_path,
        '-query', file_path,
        '-db', db_path,
        '-num_iterations', '3',
        '-evalue', '0.001',
        '-num_threads', '32',
        '-save_pssm_after_last_round',
        '-out_ascii_pssm', os.path.join(output_path, f"{name}.pssm"),
    ]

    log_path = os.path.join(output_path, 'psi.out')
    outfile = open(log_path, 'w')
    process = subprocess.run(psiblast_args, stdout=outfile, stderr=subprocess.STDOUT, close_fds = True, cwd = output_path)
    outfile.close()

if __name__ == '__main__':
    for file in os.listdir(fasta_path):
        file_path = os.path.join(fasta_path, file)
        name = os.path.splitext(os.path.basename(file))[0]

        run_psiblast(file_path, name)
    print('finish')
