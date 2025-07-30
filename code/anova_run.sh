#!/bin/bash
#SBATCH --job-name=exp4_anova         # Job name
#SBATCH --output=../log/exp4_anova_%A_%a.out  # Standard output log file
#SBATCH --error=../log/exp4_anova_%A_%a.err   # Standard error log file
#SBATCH --nodes=1                     # Number of nodes to use
#SBATCH --ntasks-per-node=40          # Number of tasks (processes) per node
#SBATCH --exclusive                   # Use the node exclusively for this job
#SBATCH --mail-type=END
#SBATCH --mail-user=cl2769@cornell.edu

# Load Python module
module load python/3.11.5

# Activate Python virtual environment
source ~/VEnvs/egg/bin/activate

# Print start message
echo "Running anova simulation for portion index $1 ..."

# Run the script with portion index passed as argument
python3 anova_run.py "$1"

# Print completion message
echo "Completed anova simulation for portion index $1"