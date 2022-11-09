# This file quickly generates the job paramters file

filename="my_job_parameters_file.sh"
walltime="2:00:00" # wall time
job_name="run_parameters_file" # slurm job name
nodes=1 # number of nodes
ntasks_per_node=16 # number of processors per node; refer to https://docs.computecanada.ca/wiki/Advanced_MPI_scheduling
memory_per_node="63G" # requested memory per node; NOTE: Must correspond to ntasks_per_node*memory_per_core; refer to https://docs.computecanada.ca/wiki/Advanced_MPI_scheduling
user_email="firstname.lastname@mail.mcgill.ca" # for receiving email notifications
compute_canada_username="username"
parameters_file="my_parameters_file.prm" # input parameter file for PHiLiP

# Dimensions of the problem
# WARNING: Must correspond to the DIM in the .prm file
PHiLiP_DIM=2

# Flag for running the code on the temporary directory
# Desription: Set as true for fast write speeds, however,
#             output files will only be copied to your job
#             submit directory once mpirun has completed. 
run_on_temp_dir=false

source ./create_job_parameters_file.sh \
${filename} \
${walltime} \
${job_name} \
${nodes} \
${ntasks_per_node} \
${user_email} \
${compute_canada_username} \
${parameters_file} \
${PHiLiP_DIM} \
${run_on_temp_dir} \
${memory_per_node}
