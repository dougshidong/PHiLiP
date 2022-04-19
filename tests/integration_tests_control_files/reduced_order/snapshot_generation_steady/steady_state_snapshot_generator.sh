#!/bin/bash

rewienski_a=(3.50000
             7.50000
             5.50000
             9.50000
             2.25000
             6.25000
             4.25000
             8.25000
             3.25000
             7.25000
             5.25000
             9.25000
             2.75000
             6.75000
             4.75000
             8.75000
             3.75000
             7.75000)

rewienski_b=(0.02333
             0.05333
             0.08333
             0.03333
             0.06333
             0.09333
             0.01667
             0.04667
             0.07667
             0.02667
             0.05667
             0.08667
             0.03667
             0.06667
             0.09667
             0.01111
             0.04111
             0.07111)

for ((i = 0 ; i < ${#rewienski_a[@]} ; i++)); do

file="${rewienski_a[i]}_${rewienski_b[i]}_1d_burgers_rewienski_snapshot_steady.prm"

echo "# Listing of Parameters"                                                                      >> $file   
echo "# ---------------------"                                                                      >> $file   
echo " "                                                                                            >> $file   
echo "set dimension = 1 "                                                                           >> $file   
echo "set test_type = finite_difference_sensitivity"                                                >> $file
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
echo " set nonlinear_max_iterations            = 50"                                               >> $file
echo " set nonlinear_steady_residual_tolerance = 1e-15"                                             >> $file
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