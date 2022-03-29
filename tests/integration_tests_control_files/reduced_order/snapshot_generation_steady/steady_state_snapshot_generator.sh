#!/bin/bash

rewienski_a=( 3.0000   3.0000     3.0000
              3.0000   3.0000     3.0000
              3.0000   3.0000     3.0000
              3.0000   3.0000     3.0000
              3.0000   3.0000     3.0000
              3.0000   3.0000     3.0000)

rewienski_b=(  0.0100    0.0153    0.0206
               0.0259    0.0312    0.0365
               0.0418    0.0471    0.0524
               0.0576    0.0629    0.0682
               0.0735    0.0788    0.0841
               0.0894    0.0947    0.1000)

for ((i = 0 ; i < ${#rewienski_a[@]} ; i++)); do

file="${rewienski_a[i]}_${rewienski_b[i]}_1d_burgers_rewienski_snapshot_steady.prm"

echo "# Listing of Parameters"                                                                      >> $file   
echo "# ---------------------"                                                                      >> $file   
echo " "                                                                                            >> $file   
echo "set dimension = 1 "                                                                           >> $file   
echo "set test_type = flow_solver"                                                                  >> $file   
echo "set pde_type = burgers_rewienski"                                                             >> $file   
echo " "                                                                                            >> $file   
echo "set use_weak_form = true"                                                                     >> $file   
echo "set use_collocated_nodes = false"                                                             >> $file   
echo " "                                                                                            >> $file   
echo "subsection grid refinement study"                                                             >> $file   
echo " set num_refinements = 10"                                                                    >> $file   
echo " set poly_degree = 0"                                                                         >> $file   
echo " set grid_left = 0.0"                                                                         >> $file   
echo " set grid_right = 100.0"                                                                      >> $file   
echo "end"                                                                                          >> $file   
echo " "                                                                                            >> $file   
echo "#Burgers parameters"                                                                          >> $file
echo "subsection burgers"                                                                           >> $file
echo " set rewienski_a = ${rewienski_a[i]}"                                                         >> $file   
echo " set rewienski_b = ${rewienski_b[i]}"                                                         >> $file   
echo "end"                                                                                          >> $file   
echo " "                                                                                            >> $file   
echo "subsection flow_solver"                                                                       >> $file   
echo " set flow_case_type = burgers_rewienski_snapshot"                                             >> $file   
echo " set final_time = 0.5"                                                                        >> $file
echo " set sensitivity_table_filename = ${rewienski_b[i]}_sensitivity_table_steady"                 >> $file
echo " set steady_state = true"                                                                     >> $file   
echo "end"                                                                                          >> $file   
echo " "                                                                                            >> $file   
echo "subsection ODE solver "                                                                       >> $file   
echo " set initial_time_step = 0.1"                                                                 >> $file   
echo " set nonlinear_max_iterations            = 500"                                               >> $file   
echo " set nonlinear_steady_residual_tolerance = 1e-12"                                             >> $file   
echo " set print_iteration_modulo              = 1"                                                 >> $file   
echo " set output_solution_vector_modulo       = 1"                                                 >> $file   
echo " set solutions_table_filename = ${rewienski_a[i]}_${rewienski_b[i]}_solutions_table_steady"   >> $file   
echo " set ode_solver_type                     = implicit"                                          >> $file   
echo " end"                                                                                         >> $file   
echo " "                                                                                            >> $file   
echo "subsection manufactured solution convergence study"                                           >> $file   
echo " set use_manufactured_source_term = true"                                                     >> $file   
echo "end"                                                                                          >> $file   


dir=$(pwd)
mpirun "-n" "1" "$HOME/Codes/PHiLiP/cmake-build-release/bin/PHiLiP_1D" "-i" "${dir}/${file}"

rm ${file}
done

echo Done!
