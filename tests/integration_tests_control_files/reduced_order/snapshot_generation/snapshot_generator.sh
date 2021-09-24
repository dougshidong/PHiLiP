#!/bin/bash

rewienski_a=(2 6 10)
rewienski_b=(0.01 0.05 0.08)

for ((i = 0 ; i < ${#rewienski_a[@]} ; i++)); do
mkdir "${rewienski_a[i]}_${rewienski_b[i]}"

cd "${rewienski_a[i]}_${rewienski_b[i]}"

file="${rewienski_a[i]}_${rewienski_b[i]}_1d_burgers_rewienski.prm"

echo "# Listing of Parameters"                                                        >> $file                                    
echo "# ---------------------"                                                        >> $file
echo " "                                                                              >> $file
echo " set test_type = burgers_rewienski_test"                                        >> $file
echo " "                                                                              >> $file
echo "# Number of dimensions"                                                         >> $file
echo " set dimension = 1"                                                             >> $file
echo " "                                                                              >> $file
echo "#The PDE we want to solve."                                                     >> $file
echo " set pde_type  = burgers_rewienski"                                             >> $file
echo " set use_weak_form = true"                                                      >> $file
echo " set use_collocated_nodes = false"                                              >> $file
echo " "                                                                              >> $file
echo "#use the grid refinement study class to generate the grid"                      >> $file
echo "subsection grid refinement study"                                               >> $file
echo " set num_refinements = 8"                                                       >> $file
echo " set poly_degree = 0"                                                           >> $file
echo " set grid_left = 0.0"                                                           >> $file
echo " set grid_right = 100.0"                                                        >> $file
echo "end"                                                                            >> $file
echo " "                                                                              >> $file
echo "#Reduced order parameters"                                                      >> $file
echo "subsection reduced order"                                                       >> $file
echo " set rewienski_a = ${rewienski_a[i]}"                                           >> $file
echo " set rewienski_b = ${rewienski_b[i]}"                                           >> $file
echo " set final_time = 5"                                                            >> $file
echo "end"                                                                            >> $file
echo " "                                                                              >> $file
echo "subsection ODE solver"                                                          >> $file
echo " set initial_time_step = 0.01"                                                  >> $file
echo " # Maximum nonlinear solver iterations"                                         >> $file
echo " set nonlinear_max_iterations            = 500"                                 >> $file
echo " "                                                                              >> $file
echo " # Nonlinear solver residual tolerance"                                         >> $file
echo " set nonlinear_steady_residual_tolerance = 1e-12"                               >> $file
echo " "                                                                              >> $file
echo " # Print every print_iteration_modulo iterations of the nonlinear solver"       >> $file
echo " set print_iteration_modulo              = 10000"                               >> $file
echo " "                                                                              >> $file
echo " # Output solution every output_solution_vector_modulo iterations in text file" >> $file
echo " set output_solution_vector_modulo        = 10"                                 >> $file
echo " "                                                                              >> $file
echo " # Explicit or implicit solverChoices are <explicit|implicit>."                 >> $file
echo " set ode_solver_type                         = implicit"                        >> $file
echo " end"                                                                           >> $file
echo " "                                                                              >> $file
echo "subsection manufactured solution convergence study"                             >> $file
echo " set use_manufactured_source_term = true"                                       >> $file
echo "end"                                                                            >> $file

/usr/bin/mpirun "-n" "1" "$HOME/Codes/PHiLiP/cmake-build-release/bin/PHiLiP_1D" "-i" "$HOME/Codes/PHiLiP/cmake-build-release/tests/integration_tests_control_files/reduced_order/snapshot_generation/${rewienski_a[i]}_${rewienski_b[i]}/${rewienski_a[i]}_${rewienski_b[i]}_1d_burgers_rewienski.prm"

cd ..
done

echo Done!