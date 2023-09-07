#ifndef __DG_HELPER_FUNCTIONS_HPP__
#define __DG_HELPER_FUNCTIONS_HPP__

template <int dim, typename real>
inline real determinant_ArrayTensor(std::array<dealii::Tensor<1,dim,real>,dim> &array_tensor)
{
    if constexpr(dim==1) {
        return array_tensor[0][0];
    } else if constexpr(dim==2) {
        return (array_tensor[0][0] * array_tensor[1][1] - array_tensor[0][1] * array_tensor[1][0]);
    } else if constexpr(dim==3) {
        return (array_tensor[0][0] * (array_tensor[1][1] * array_tensor[2][2] - array_tensor[1][2] * array_tensor[2][1])
               -array_tensor[0][1] * (array_tensor[1][0] * array_tensor[2][2] - array_tensor[1][2] * array_tensor[2][0])
               +array_tensor[0][2] * (array_tensor[1][0] * array_tensor[2][1] - array_tensor[1][1] * array_tensor[2][0]));
    }
}


template <int dim, typename real, int n_components>
void evaluate_finite_element_values (
    const std::vector<dealii::Point<dim>> &unit_points,
    const std::vector<real> &coefficients,
    const dealii::FESystem<dim,dim> &finite_element,
    std::vector< std::array<real,n_components> > &values)
{
    const unsigned int n_dofs = finite_element.dofs_per_cell;
    const unsigned int n_pts = unit_points.size();

    AssertDimension(n_dofs, coefficients.size());

    for (unsigned int ipoint=0; ipoint<n_pts; ++ipoint) {
        for (int icomp=0; icomp<n_components; ++icomp) {
            values[ipoint][icomp] = 0;
        }
        for (unsigned int idof = 0; idof < n_dofs; ++idof) {
            const int icomp = finite_element.system_to_component_index(idof).first;
            values[ipoint][icomp] += coefficients[idof] * finite_element.shape_value_component(idof, unit_points[ipoint], icomp);
        }
    }
}



template <int dim, typename real, int n_components>
void evaluate_finite_element_gradients (
    const std::vector<dealii::Point<dim>> &unit_points,
    const std::vector<real> &coefficients,
    const dealii::FESystem<dim,dim> &finite_element,
    std::vector < std::array< dealii::Tensor<1,dim,real>, n_components > > &gradients)
{
    AssertDimension(unit_points.size(), gradients.size());
    const unsigned int n_dofs = finite_element.dofs_per_cell;
    const unsigned int n_pts = unit_points.size();

    AssertDimension(n_dofs, coefficients.size());
    AssertDimension(finite_element.n_components(), n_components);

    for (unsigned int ipoint=0; ipoint<n_pts; ++ipoint) {
        for (int icomp=0; icomp<n_components; ++icomp) {
            gradients[ipoint][icomp] = 0;
        }
        for (unsigned int idof = 0; idof < n_dofs; ++idof) {
            const int icomp = finite_element.system_to_component_index(idof).first;
            dealii::Tensor<1,dim,double> shape_grad = finite_element.shape_grad_component (idof, unit_points[ipoint], icomp);
            for (int d=0; d<dim; ++d) {
                gradients[ipoint][icomp][d] += coefficients[idof] * shape_grad[d];
            }
        }
    }
}


template <int dim, typename real>
void evaluate_covariant_metric_jacobian (
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<real> &coords_coeff,
    const dealii::FESystem<dim,dim> &fe_metric,
    std::vector<dealii::Tensor<2,dim,real>> &covariant_metric_jacobian,
    std::vector<real> &jacobian_determinants)
{
    const std::vector< dealii::Point<dim,double> > &unit_quad_pts = quadrature.get_points();
    const unsigned int n_quad_pts = unit_quad_pts.size();

    //const unsigned int grid_degree = fe_metric.tensor_degree();
    //const dealii::FE_Q<dim> fe_lagrange_grid(2*grid_degree);
    const dealii::FiniteElement<dim> &fe_lagrange_grid = fe_metric.base_element(0);
    const std::vector< dealii::Point<dim,double> > &unit_grid_pts = fe_lagrange_grid.get_unit_support_points();
    const unsigned int n_grid_pts = unit_grid_pts.size();

    std::vector < std::array< real,dim> > coords(n_grid_pts);
    evaluate_finite_element_values  <dim, real, dim> (unit_grid_pts, coords_coeff, fe_metric, coords);

    std::vector < std::array< dealii::Tensor<1,dim,real>, dim > > coords_gradients(n_grid_pts);
    evaluate_finite_element_gradients <dim, real, dim> (unit_grid_pts, coords_coeff, fe_metric, coords_gradients);

    std::vector < std::array< dealii::Tensor<1,dim,real>, dim > > quad_pts_coords_gradients(n_quad_pts);
    evaluate_finite_element_gradients <dim, real, dim> (unit_quad_pts, coords_coeff, fe_metric, quad_pts_coords_gradients);

    for (unsigned int iquad = 0; iquad<n_quad_pts; ++iquad) {
        jacobian_determinants[iquad] = determinant_ArrayTensor<dim,real>(quad_pts_coords_gradients[iquad]);
    }

    if constexpr (dim==1) {
        for (unsigned int iquad = 0; iquad<n_quad_pts; ++iquad) {
            const real invJ = 1.0/jacobian_determinants[iquad];
            covariant_metric_jacobian[iquad][0][0] = invJ;
        }
    }

    if constexpr (dim==2) {
        // Remark 5 of Kopriva (2006).
        // Need to interpolate physical coordinates, and then differentiate it
        // using the derivatives of the collocated Lagrange basis.

        std::vector<dealii::Tensor<2,dim,real>> dphys_dref_quad(n_quad_pts);

        // In 2D Cross-Product Form = Conservative-Curl Form
        for (unsigned int iquad = 0; iquad<n_quad_pts; ++iquad) {

            dphys_dref_quad[iquad] = 0.0;

            const dealii::Point<dim,double> &quad_point = unit_quad_pts[iquad];

            for (unsigned int igrid = 0; igrid<n_grid_pts; ++igrid) {

                const dealii::Tensor<1,dim,double> shape_grad = fe_lagrange_grid.shape_grad(igrid, quad_point);

                for(int dphys=0; dphys<dim; dphys++) {
                    for(int dref=0; dref<dim; dref++) {
                        dphys_dref_quad[iquad][dphys][dref] += coords[igrid][dphys] * shape_grad[dref];
                    }
                }
            }
        }

        // In 2D Cross-Product Form = Conservative-Curl Form
        for (unsigned int iquad = 0; iquad<n_quad_pts; ++iquad) {

            const real invJ = 1.0/jacobian_determinants[iquad];

            covariant_metric_jacobian[iquad] = 0.0;

            // inv(A)^T =  [ a  b ]^-T  =  (1/det(A)) [ d -c ]
            //             [ c  d ]                   [-b  a ]
            covariant_metric_jacobian[iquad][0][0] =  dphys_dref_quad[iquad][1][1] * invJ;
            covariant_metric_jacobian[iquad][0][1] = -dphys_dref_quad[iquad][1][0] * invJ;
            covariant_metric_jacobian[iquad][1][0] = -dphys_dref_quad[iquad][0][1] * invJ;
            covariant_metric_jacobian[iquad][1][1] =  dphys_dref_quad[iquad][0][0] * invJ;

        }

    }
    if constexpr (dim == 3) {

        // Evaluate the physical (Y grad Z), (Z grad X), (X grad
        std::vector<real> Ta(n_grid_pts); 
        std::vector<real> Tb(n_grid_pts); 
        std::vector<real> Tc(n_grid_pts);

        std::vector<real> Td(n_grid_pts);
        std::vector<real> Te(n_grid_pts);
        std::vector<real> Tf(n_grid_pts);

        std::vector<real> Tg(n_grid_pts);
        std::vector<real> Th(n_grid_pts);
        std::vector<real> Ti(n_grid_pts);

        for(unsigned int igrid=0; igrid<n_grid_pts; igrid++) {
            Ta[igrid] = 0.5*(coords_gradients[igrid][1][1] * coords[igrid][2] - coords_gradients[igrid][2][1] * coords[igrid][1]);
            Tb[igrid] = 0.5*(coords_gradients[igrid][1][2] * coords[igrid][2] - coords_gradients[igrid][2][2] * coords[igrid][1]);
            Tc[igrid] = 0.5*(coords_gradients[igrid][1][0] * coords[igrid][2] - coords_gradients[igrid][2][0] * coords[igrid][1]);

            Td[igrid] = 0.5*(coords_gradients[igrid][2][1] * coords[igrid][0] - coords_gradients[igrid][0][1] * coords[igrid][2]);
            Te[igrid] = 0.5*(coords_gradients[igrid][2][2] * coords[igrid][0] - coords_gradients[igrid][0][2] * coords[igrid][2]);
            Tf[igrid] = 0.5*(coords_gradients[igrid][2][0] * coords[igrid][0] - coords_gradients[igrid][0][0] * coords[igrid][2]);

            Tg[igrid] = 0.5*(coords_gradients[igrid][0][1] * coords[igrid][1] - coords_gradients[igrid][1][1] * coords[igrid][0]);
            Th[igrid] = 0.5*(coords_gradients[igrid][0][2] * coords[igrid][1] - coords_gradients[igrid][1][2] * coords[igrid][0]);
            Ti[igrid] = 0.5*(coords_gradients[igrid][0][0] * coords[igrid][1] - coords_gradients[igrid][1][0] * coords[igrid][0]);
        }

        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++) {

            covariant_metric_jacobian[iquad] = 0.0;

            const dealii::Point<dim,double> &quad_point  = unit_quad_pts[iquad];

            for(unsigned int igrid=0; igrid<n_grid_pts; igrid++) {

                const dealii::Tensor<1,dim,double> shape_grad = fe_lagrange_grid.shape_grad(igrid, quad_point);

                covariant_metric_jacobian[iquad][0][0] += shape_grad[2] * Ta[igrid] - shape_grad[1] * Tb[igrid];
                covariant_metric_jacobian[iquad][1][0] += shape_grad[2] * Td[igrid] - shape_grad[1] * Te[igrid];
                covariant_metric_jacobian[iquad][2][0] += shape_grad[2] * Tg[igrid] - shape_grad[1] * Th[igrid];

                covariant_metric_jacobian[iquad][0][1] += shape_grad[0] * Tb[igrid] - shape_grad[2] * Tc[igrid];
                covariant_metric_jacobian[iquad][1][1] += shape_grad[0] * Te[igrid] - shape_grad[2] * Tf[igrid];
                covariant_metric_jacobian[iquad][2][1] += shape_grad[0] * Th[igrid] - shape_grad[2] * Ti[igrid];

                covariant_metric_jacobian[iquad][0][2] += shape_grad[1] * Tc[igrid] - shape_grad[0] * Ta[igrid];
                covariant_metric_jacobian[iquad][1][2] += shape_grad[1] * Tf[igrid] - shape_grad[0] * Td[igrid];
                covariant_metric_jacobian[iquad][2][2] += shape_grad[1] * Ti[igrid] - shape_grad[0] * Tg[igrid];
            }

            const real invJ = 1.0/jacobian_determinants[iquad];
            covariant_metric_jacobian[iquad] *= invJ;

        }

    }

}

/// Code taken directly from deal.II's FullMatrix::gauss_jordan function, but adapted to
/// handle AD variable.
template <typename number>
void gauss_jordan(dealii::FullMatrix<number> &input_matrix)
{
    Assert(!input_matrix.empty(), dealii::ExcMessage("Empty matrix"))
    Assert(input_matrix.n_cols() == input_matrix.n_rows(), dealii::ExcMessage("Non quadratic matrix"));
  
    // Gauss-Jordan-Algorithm from Stoer & Bulirsch I (4th Edition) p. 153
    const size_t N = input_matrix.n();
  
    // First get an estimate of the size of the elements of this matrix,
    // for later checks whether the pivot element is large enough,
    // for whether we have to fear that the matrix is not regular
    number diagonal_sum = 0;
    for (size_t i = 0; i < N; ++i)
        diagonal_sum = diagonal_sum + abs(input_matrix(i, i));
    const number typical_diagonal_element = diagonal_sum / N;
    (void)typical_diagonal_element;
  
    // initialize the array that holds the permutations that we find during pivot search
    std::vector<size_t> p(N);
    for (size_t i = 0; i < N; ++i)
        p[i] = i;
  
    for (size_t j = 0; j < N; ++j) {
        // pivot search: search that part of the line on and
        // right of the diagonal for the largest element
        number max_pivot = abs(input_matrix(j, j));
        size_t r   = j;
        for (size_t i = j + 1; i < N; ++i) {
            if (abs(input_matrix(i, j)) > max_pivot) {
                max_pivot = abs(input_matrix(i, j));
                r   = i;
            }
        }
        // check whether the pivot is too small
        Assert(max_pivot > 1.e-16 * typical_diagonal_element, dealii::ExcMessage("Non regular matrix"));
  
        // row interchange
        if (r > j) {
            for (size_t k = 0; k < N; ++k)
                std::swap(input_matrix(j, k), input_matrix(r, k));
  
            std::swap(p[j], p[r]);
        }
  
        // transformation
        const number hr = number(1.) / input_matrix(j, j);
        input_matrix(j, j)   = hr;
        for (size_t k = 0; k < N; ++k) {
            if (k == j) continue;
            for (size_t i = 0; i < N; ++i) {
                if (i == j) continue;
                input_matrix(i, k) -= input_matrix(i, j) * input_matrix(j, k) * hr;
            }
        }
        for (size_t i = 0; i < N; ++i) {
            input_matrix(i, j) *= hr;
            input_matrix(j, i) *= -hr;
        }
        input_matrix(j, j) = hr;
    }
    // column interchange
    std::vector<number> hv(N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < N; ++k)
            hv[p[k]] = input_matrix(i, k);
        for (size_t k = 0; k < N; ++k)
            input_matrix(i, k) = hv[k];
    }
}


template<int dim, typename real>
std::vector< real > project_function(
    const std::vector< real > &function_coeff,
    const dealii::FESystem<dim,dim> &fe_input,
    const dealii::FESystem<dim,dim> &fe_output,
    const std::vector< real > &coords_coeff,
    const dealii::FESystem<dim,dim> &fe_metric,
    const dealii::QGauss<dim> &projection_quadrature)
{
    const unsigned int nstate = fe_input.n_components();
    const unsigned int n_vector_dofs_in = fe_input.dofs_per_cell;
    const unsigned int n_vector_dofs_out = fe_output.dofs_per_cell;
    const unsigned int n_dofs_in = n_vector_dofs_in / nstate;
    const unsigned int n_dofs_out = n_vector_dofs_out / nstate;

    assert(n_vector_dofs_in == function_coeff.size());
    assert(nstate == fe_output.n_components());

    const unsigned int n_quad_pts = projection_quadrature.size();
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = projection_quadrature.get_points();

    std::vector< real > function_coeff_out(n_vector_dofs_out); // output function coefficients.

    std::vector<dealii::Tensor<2,dim,real>> covariant_metric_jacobian(n_quad_pts);
    std::vector<real> jacobian_determinants(n_quad_pts);
    std::vector<real> jxw(n_quad_pts);
    evaluate_covariant_metric_jacobian(projection_quadrature, coords_coeff, fe_metric, covariant_metric_jacobian, jacobian_determinants);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        jxw[iquad] = jacobian_determinants[iquad] * projection_quadrature.weight(iquad);
    }

    for (unsigned istate = 0; istate < nstate; ++istate) {

        std::vector< real > function_at_quad(n_quad_pts);

        // Output interpolation_operator is V^T in the notes.
        dealii::FullMatrix<real> interpolation_operator(n_dofs_out,n_quad_pts);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            function_at_quad[iquad] = 0.0;
            for (unsigned int idof=0; idof<n_dofs_in; ++idof) {
                const unsigned int idof_vector = fe_input.component_to_system_index(istate,idof);
                function_at_quad[iquad] += function_coeff[idof_vector] * fe_input.shape_value_component(idof_vector,unit_quad_pts[iquad],istate);
            }

            for (unsigned int idof=0; idof<n_dofs_out; ++idof) {
                const unsigned int idof_vector = fe_output.component_to_system_index(istate,idof);
                interpolation_operator[idof][iquad] = fe_output.shape_value_component(idof_vector,unit_quad_pts[iquad],istate);
            }
        }

        std::vector< real > rhs(n_dofs_out);
        for (unsigned int idof=0; idof<n_dofs_out; ++idof) {
            rhs[idof] = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                rhs[idof] += interpolation_operator[idof][iquad] * function_at_quad[iquad] * jxw[iquad];
            }
        }

        dealii::FullMatrix<real> mass(n_dofs_out, n_dofs_out);
        for(unsigned int row=0; row<n_dofs_out; ++row) {
            for(unsigned int col=0; col<n_dofs_out; ++col) {
                mass[row][col] = 0;
            }
        }
        for(unsigned int row=0; row<n_dofs_out; ++row) {
            for(unsigned int col=0; col<n_dofs_out; ++col) {
                for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    mass[row][col] += interpolation_operator[row][iquad] * interpolation_operator[col][iquad] * jxw[iquad];
                }
            }
        }
        dealii::FullMatrix<real> inverse_mass(n_dofs_out, n_dofs_out);
        inverse_mass = mass;
        gauss_jordan(inverse_mass);

        for(unsigned int row=0; row<n_dofs_out; ++row) {
            const unsigned int idof_vector = fe_output.component_to_system_index(istate,row);
            function_coeff_out[idof_vector] = 0.0;
            for(unsigned int col=0; col<n_dofs_out; ++col) {
                function_coeff_out[idof_vector] += inverse_mass[row][col] * rhs[col];
            }
        }
    }

    return function_coeff_out;
}

#endif
