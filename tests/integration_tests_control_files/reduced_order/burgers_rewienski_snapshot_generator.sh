#!/bin/bash

rewienski_a=(2
             2
             10
             10
             6
             5.54268848508671)

rewienski_b=(0.01
             0.1
             0.1
             0.01
             0.055
             0.1)

for ((i = 0 ; i < ${#rewienski_a[@]} ; i++)); do

file="${rewienski_a[i]}_${rewienski_b[i]}_1d_burgers_rewienski_snapshot.prm"

echo "# Listing of Parameters"                                                                      >> $file
echo "# ---------------------"                                                                      >> $file
echo " "                                                                                            >> $file
echo "set dimension = 1 "                                                                           >> $file
echo "set run_type = flow_simulation"                                                               >> $file
echo "set pde_type = burgers_rewienski"                                                             >> $file
echo " "                                                                                            >> $file
echo "set use_weak_form = true"                                                                     >> $file
echo "set use_collocated_nodes = false"                                                             >> $file
echo " "                                                                                            >> $file
echo "subsection grid refinement study"                                                             >> $file
echo " set num_refinements = 10"                                                                    >> $file
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
echo " set steady_state = true"                                                                     >> $file
echo " set poly_degree = 0"                                                                         >> $file
echo "  subsection grid"                                                                            >> $file
echo "   set grid_left_bound = 0.0"                                                                       >> $file
echo "   set grid_right_bound = 100.0"                                                                    >> $file
echo "  end"                                                                                        >> $file
echo "end"                                                                                          >> $file
echo " "                                                                                            >> $file
echo "subsection ODE solver "                                                                       >> $file
echo " set nonlinear_max_iterations            = 50"                                               >> $file
echo " set nonlinear_steady_residual_tolerance = 1e-15"                                             >> $file
echo " set print_iteration_modulo              = 1"                                                 >> $file
echo " set output_final_steady_state_solution_to_file       = true"                                 >> $file
echo " set steady_state_final_solution_filename = ${rewienski_a[i]}_${rewienski_b[i]}_solution_snapshot">> $file
echo " set ode_solver_type                     = implicit"                                          >> $file
echo " end"                                                                                         >> $file
echo " "                                                                                            >> $file
echo "subsection manufactured solution convergence study"                                           >> $file
echo " set use_manufactured_source_term = true"                                                     >> $file
echo "end"                                                                                          >> $file


dir=$(pwd)
mpirun "-n" "1" "$1/PHiLiP_1D" "-i" "${dir}/${file}"

rm ${file}
done

echo Done!