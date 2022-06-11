#!/bin/bash

mach=(0.5
      0.9
      0.9
      0.7
      0.5
      0.7
      0.6
      0.8
      0.55
      0.75
      0.65
      0.85
      0.525
      0.725
      0.625
      0.825
      0.575
      0.775
      0.675
      0.875
      0.5125
      0.7125
      0.6125
      0.8125
      0.5625
      0.7625
      0.6625
      0.8625
      0.5375
      0.9)

alpha=(4
       4
       0
       2
       0
       1.33333333333333
       2.66666666666667
       0.444444444444446
       1.77777777777778
       3.11111111111111
       0.888888888888892
       2.22222222222222
       3.55555555555555
       0.148148148148147
       1.48148148148148
       2.81481481481482
       0.592592592592593
       1.92592592592592
       3.25925925925926
       1.03703703703704
       2.37037037037037
       3.7037037037037
       0.296296296296299
       1.62962962962963
       2.96296296296296
       0.740740740740739
       2.07407407407408
       3.40740740740741
       1.18518518518519
       0.534455260634153)

for ((i = 0 ; i < ${#mach[@]} ; i++)); do

file="${mach[i]}_${alpha[i]}_naca0012.prm"

echo "set test_type = flow_solver" >> $file
echo "set dimension = 2" >> $file
echo "set pde_type  = euler" >> $file
echo "" >> $file
echo "set conv_num_flux = roe" >> $file
echo "set diss_num_flux = bassi_rebay_2" >> $file
echo "" >> $file
echo "set use_split_form = false" >> $file
echo "" >> $file
echo "subsection artificial dissipation" >> $file
echo "	set add_artificial_dissipation = true" >> $file
echo "end" >> $file
echo "" >> $file
echo "set overintegration = 0" >> $file
echo "" >> $file
echo "subsection euler" >> $file
echo "  set reference_length = 1.0" >> $file
echo "  set mach_infinity = ${mach[i]}" >> $file
echo "  set angle_of_attack = ${alpha[i]}" >> $file
echo "end" >> $file
echo "" >> $file
echo "subsection linear solver" >> $file
echo "  subsection gmres options" >> $file
echo "    set ilut_atol                 = 1e-4" >> $file
echo "    set ilut_rtol                 = 1.00001" >> $file
echo "    set ilut_drop                 = 0.0" >> $file
echo "    set ilut_fill                 = 10" >> $file
echo "    set linear_residual_tolerance = 1e-13" >> $file
echo "    set max_iterations            = 2000" >> $file
echo "    set restart_number            = 200" >> $file
echo "  end" >> $file
echo "end" >> $file
echo "" >> $file
echo "subsection ODE solver" >> $file
echo "  set output_solution_every_x_steps = 1" >> $file
echo "  set nonlinear_max_iterations            = 2000" >> $file
echo "  set nonlinear_steady_residual_tolerance = 1e-15" >> $file
echo "  set ode_solver_type  = implicit" >> $file
echo "  set initial_time_step = 1e3" >> $file
echo "  set time_step_factor_residual = 15.0" >> $file
echo "  set time_step_factor_residual_exp = 2" >> $file
echo "  #set print_iteration_modulo              = 1" >> $file
echo "  set output_solution_vector_modulo       = 1" >> $file
echo "  set solutions_table_filename = ${mach[i]}_${alpha[i]}_solution_snapshot" >> $file
echo "end" >> $file
echo "" >> $file
echo "subsection grid refinement study" >> $file
echo " set poly_degree = 0" >> $file
echo " set num_refinements = 0" >> $file
echo "end" >> $file
echo "" >> $file
echo "subsection flow_solver" >> $file
echo "  set flow_case_type = naca0012" >> $file
echo "  set input_mesh_filename = naca0012_hopw_ref1" >> $file
echo "  set steady_state = true" >> $file
echo "end" >> $file

dir=$(pwd)
mpirun "-n" "4" "$1/PHiLiP_2D" "-i" "${dir}/${file}"

rm ${file}
done

echo Done!