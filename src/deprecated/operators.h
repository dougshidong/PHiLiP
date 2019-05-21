    template <int dim, typename real>
    void PDE<dim, real>::compute_inv_mass_matrix()
    {
        unsigned int fe_degree = fe.get_degree();
        QGauss<dim> quadrature(fe_degree+1);
        unsigned int n_quad_pts = quadrature.size();
        FEValues<dim> fe_values(mapping, fe, quadrature, update_values | update_JxW_values);
        
        std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
        
        typename DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active(),
           endc = dof_handler.end();
        
        // Allocate inverse mass matrices
        inv_mass_matrix.resize(triangulation.n_active_cells(),
                               FullMatrix<real>(fe.dofs_per_cell));

        for (; cell!=endc; ++cell) {

            const unsigned int icell = cell->user_index();
            cell->get_dof_indices (dof_indices);
            fe_values.reinit(cell);

            for(unsigned int idof=0; idof<fe.dofs_per_cell; ++idof) {
            for(unsigned int jdof=0; jdof<fe.dofs_per_cell; ++jdof) {

                inv_mass_matrix[icell][idof][jdof] = 0.0;

                for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    inv_mass_matrix[icell][idof][jdof] +=
                        fe_values.shape_value(idof,iquad) *
                        fe_values.shape_value(jdof,iquad) *
                        fe_values.JxW(iquad);
                }
            }
            }

            // Invert mass matrix
            inv_mass_matrix[icell].gauss_jordan();
        }

    }

    template <int dim, typename real>
    void PDE<dim, real>::compute_stiffness_matrix()
    {
        unsigned int fe_degree = fe.get_degree();
        QGauss<dim> quadrature(fe_degree+1);
        unsigned int n_quad_pts = quadrature.size();
        FEValues<dim> fe_values(mapping, fe, quadrature, update_values | update_JxW_values);
        
        std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
        
        typename DoFHandler<dim>::active_cell_iterator
           cell = dof_handler.begin_active(),
           endc = dof_handler.end();
        
        // Allocate inverse mass matrices
        inv_mass_matrix.resize(triangulation.n_active_cells(),
                               FullMatrix<real>(fe.dofs_per_cell));

        for (; cell!=endc; ++cell) {

            const unsigned int icell = cell->user_index();
            cell->get_dof_indices (dof_indices);
            fe_values.reinit(cell);

            for(unsigned int idof=0; idof<fe.dofs_per_cell; ++idof) {
            for(unsigned int jdof=0; jdof<fe.dofs_per_cell; ++jdof) {

                inv_mass_matrix[icell][idof][jdof] = 0.0;

                for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    inv_mass_matrix[icell][idof][jdof] +=
                        fe_values.shape_value(idof,iquad) *
                        fe_values.shape_value(jdof,iquad) *
                        fe_values.JxW(iquad);
                }
            }
            }

            // Invert mass matrix
            inv_mass_matrix[icell].gauss_jordan();
        }

    }

