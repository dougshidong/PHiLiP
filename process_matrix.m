close all;
m = size(symm_matrix)
n_cells = 140
dg_poly = 1
dim = 2
n_state = dim+2
n_des = 20
n_constraints = 3

n_dofs_per_cell = (dg_poly + 1)^dim * n_state
n_dofs = n_dofs_per_cell * n_cells

n_sim = n_dofs
n_des = n_des
n_slack = n_constraints
n_sim_dual = n_dofs
n_constraints_dual = n_constraints
n_slack_dual = (n_sim+ n_des + n_slack)

total = n_sim + n_des + n_slack + n_sim_dual + n_constraints_dual + n_slack_dual

symmetrize = symm_matrix;

s_state = 1;         e_state = n_dofs;
s_desig = e_state+1; e_desig = s_desig + n_des - 1;
s_slack = e_desig+1; e_slack = s_slack + n_slack - 1;
s_resid = e_slack+1; e_resid = s_resid + n_dofs - 1;
s_const = e_resid+1; e_const = s_const + n_constraints - 1;
s_inequ = e_const+1; e_inequ = s_inequ + n_slack_dual - 1;

symmetrize(s_state:e_slack,s_inequ:e_inequ) = 0.0;
symmetrize(s_const:e_inequ,s_const:e_inequ) = 0.0;
figure; spy(symm_matrix); figure; spy(symmetrize)
%figure; spy(symm_matrix - symmetrize)
%spy(symm_matrix(s_state:e_state,s_state:e_state))
symmetrize(1:n_sim,s_resid:e_resid) = eye(n_dofs);
symmetrize(s_resid:e_resid,s_state:e_state) = eye(n_dofs);spy(symmetrize)