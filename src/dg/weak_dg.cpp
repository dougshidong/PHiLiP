#include <deal.II/base/tensor.h>
#include <deal.II/base/table.h>

#include <deal.II/base/qprojector.h>

#include <deal.II/lac/full_matrix.templates.h>
//#include <deal.II/lac/full_matrix.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include "ADTypes.hpp"

#include "weak_dg.hpp"

#define KOPRIVA_METRICS_VOL
#define KOPRIVA_METRICS_FACE
#define KOPRIVA_METRICS_BOUNDARY
//#define FADFAD

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

/// Returns the value from a CoDiPack variable.
/** The recursive calling allows to retrieve nested CoDiPack types.
 */
template<typename real>
double getValue(const real &x) {
    if constexpr(std::is_same<real,double>::value) {
        return x;
    } else {
        return getValue(x.value());
    }
}
/// Returns y = Ax.
/** Had to rewrite this instead of 
 *  dealii::contract<1,0>(A,x);
 *  because contract doesn't allow the use of codi variables.
 */
template<int dim, typename real1, typename real2>
dealii::Tensor<1,dim,real1> vmult(const dealii::Tensor<2,dim,real1> A, const dealii::Tensor<1,dim,real2> x)
{
     dealii::Tensor<1,dim,real1> y;
     for (int row=0;row<dim;++row) {
         y[row] = 0.0;
         for (int col=0;col<dim;++col) {
             y[row] += A[row][col] * x[col];
         }
     }
     return y;
}

/// Returns norm of dealii::Tensor<1,dim,real>
/** Had to rewrite this instead of 
 *  x.norm()
 *  because norm() doesn't allow the use of codi variables.
 */
template<int dim, typename real1>
real1 norm(const dealii::Tensor<1,dim,real1> x)
{
     real1 val = 0.0;
     for (int row=0;row<dim;++row) {
         val += x[row] * x[row];
     }
     return sqrt(val);
}

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
DGWeak<dim,nstate,real,MeshType>::DGWeak(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBaseState<dim,nstate,real,MeshType>::DGBaseState(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
{ }
// Destructor
template <int dim, int nstate, typename real,typename MeshType>
DGWeak<dim,nstate,real,MeshType>::~DGWeak ()
{
    pcout << "Destructing DGWeak..." << std::endl;
}


/// Derivative indexing when only 1 cell is concerned.
/// Derivatives are ordered such that w comes first with index 0, then x.
/// If derivatives with respect to w are not needed, then derivatives
/// with respect to x will start at index 0. This function is for a single
/// cell's DoFs.
void automatic_differentiation_indexing_1(
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
    const unsigned int n_soln_dofs, const unsigned int n_metric_dofs,
    unsigned int &w_start, unsigned int &w_end,
    unsigned int &x_start, unsigned int &x_end)
{
    w_start = 0;
    w_end = 0;
    x_start = 0;
    x_end = 0;
    if (compute_d2R || (compute_dRdW && compute_dRdX)) {
        w_start = 0;
        w_end   = w_start + n_soln_dofs;
        x_start = w_end;
        x_end   = x_start + n_metric_dofs;
    } else if (compute_dRdW) {
        w_start = 0;
        w_end   = w_start + n_soln_dofs;
        x_start = w_end;
        x_end   = x_start + 0;
    } else if (compute_dRdX) {
        w_start = 0;
        w_end   = w_start + 0;
        x_start = w_end;
        x_end   = x_start + n_metric_dofs;
    } else {
        std::cout << "Called the derivative version of the residual without requesting the derivative" << std::endl;
    }
}

/// Derivative indexing when 2 cells are concerned.
/// Derivatives are ordered such that w comes first with index 0, then x.
/// If derivatives with respect to w are not needed, then derivatives
/// with respect to x will start at index 0. This function is for a single
/// cell's DoFs.
void automatic_differentiation_indexing_2(
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
    const unsigned int n_soln_dofs_int, const unsigned int n_soln_dofs_ext, const unsigned int n_metric_dofs,
    unsigned int &w_int_start, unsigned int &w_int_end, unsigned int &w_ext_start, unsigned int &w_ext_end,
    unsigned int &x_int_start, unsigned int &x_int_end, unsigned int &x_ext_start, unsigned int &x_ext_end)
{
    // Current derivative order is: soln_int, soln_ext, metric_int, metric_ext
    w_int_start = 0; w_int_end = 0; w_ext_start = 0; w_ext_end = 0;
    x_int_start = 0; x_int_end = 0; x_ext_start = 0; x_ext_end = 0;
    if (compute_d2R || (compute_dRdW && compute_dRdX)) {
        w_int_start = 0;
        w_int_end   = w_int_start + n_soln_dofs_int;
        w_ext_start = w_int_end;
        w_ext_end   = w_ext_start + n_soln_dofs_ext;
        x_int_start = w_ext_end;
        x_int_end   = x_int_start + n_metric_dofs;
        x_ext_start = x_int_end;
        x_ext_end   = x_ext_start + n_metric_dofs;
    } else if (compute_dRdW) {
        w_int_start = 0;
        w_int_end   = w_int_start + n_soln_dofs_int;
        w_ext_start = w_int_end;
        w_ext_end   = w_ext_start + n_soln_dofs_ext;
        x_int_start = w_ext_end;
        x_int_end   = x_int_start + 0;
        x_ext_start = x_int_end;
        x_ext_end   = x_ext_start + 0;
    } else if (compute_dRdX) {
        w_int_start = 0;
        w_int_end   = w_int_start + 0;
        w_ext_start = w_int_end;
        w_ext_end   = w_ext_start + 0;
        x_int_start = w_ext_end;
        x_int_end   = x_int_start + n_metric_dofs;
        x_ext_start = x_int_end;
        x_ext_end   = x_ext_start + n_metric_dofs;
    } else {
        std::cout << "Called the derivative version of the residual without requesting the derivative" << std::endl;
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


template <int dim, typename real>
bool check_same_coords (
    const std::vector<dealii::Point<dim>> &unit_quad_pts_int,
    const std::vector<dealii::Point<dim>> &unit_quad_pts_ext,
    const std::vector<real> &coords_coeff_int,
    const std::vector<real> &coords_coeff_ext,
    const dealii::FESystem<dim,dim> &fe_metric,
    const double tolerance)
{
    assert(unit_quad_pts_int.size() == unit_quad_pts_ext.size());
    const unsigned int nquad = unit_quad_pts_int.size();
    std::vector < std::array< real,dim> > coords_int(nquad);
    std::vector < std::array< real,dim> > coords_ext(nquad);
    evaluate_finite_element_values  <dim, real, dim> (unit_quad_pts_int, coords_coeff_int, fe_metric, coords_int);
    evaluate_finite_element_values  <dim, real, dim> (unit_quad_pts_ext, coords_coeff_ext, fe_metric, coords_ext);

    bool issame = true;
    for (unsigned int iquad = 0; iquad < nquad; ++iquad) {
        for (int d=0; d<dim; ++d) {
            real abs_diff = abs(coords_int[iquad][d] - coords_ext[iquad][d]);
            if (abs_diff > tolerance) {
                real rel_diff = abs_diff / coords_int[iquad][d];
                if (rel_diff > tolerance) {
                    issame = false;
                }
            }
        }
        if (!issame) {
            std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
            std::cout << "coords_int ";
            for (int d=0;d<dim;++d) {
                std::cout << coords_int[iquad][d] << " ";
            }
            std::cout << std::endl;
            std::cout << "coords_ext ";
            for (int d=0;d<dim;++d) {
                std::cout << coords_ext[iquad][d] << " ";
            }
            std::cout << std::endl;
        }
    }
    return issame;
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
std::vector<dealii::Tensor<2,dim,real>> evaluate_metric_jacobian (
    const std::vector<dealii::Point<dim>> &points,
    const std::vector<real> &coords_coeff,
    const dealii::FESystem<dim,dim> &fe_metric)
{
    const unsigned int n_dofs = fe_metric.dofs_per_cell;
    (void) n_dofs;
    const unsigned int n_pts = points.size();

    AssertDimension(n_dofs, coords_coeff.size());

    std::vector < std::array< dealii::Tensor<1,dim,real>, dim > > coords_gradients(n_pts);
    evaluate_finite_element_gradients<dim, real, dim> (points, coords_coeff, fe_metric, coords_gradients);

    std::vector<dealii::Tensor<2,dim,real>> metric_jacobian(n_pts);

    for (unsigned int ipoint=0; ipoint<n_pts; ++ipoint) {
        for (int row=0;row<dim;++row) {
            for (int col=0;col<dim;++col) {
                metric_jacobian[ipoint][row][col] = coords_gradients[ipoint][row][col];
            }
        }
    }
    return metric_jacobian;
}

template <int dim, typename real>
std::vector <real> determinant_ArrayTensor(std::vector < std::array< dealii::Tensor<1,dim,real>, dim > > &coords_gradients)
{
    const unsigned int n = coords_gradients.size();
    std::vector <real> determinants(n);
    for (unsigned int i=0; i<n; ++i) {
        if constexpr(dim==1) {
            determinants[i] =  coords_gradients[i][0][0];
        }
        if constexpr(dim==2) {
            determinants[i] =  coords_gradients[i][0][0] * coords_gradients[i][1][1] - coords_gradients[i][0][1] * coords_gradients[i][1][0];
        }
        if constexpr(dim==3) {
            determinants[i] = +coords_gradients[i][0][0] * (coords_gradients[i][1][1] * coords_gradients[i][2][2] - coords_gradients[i][1][2] * coords_gradients[i][2][1])
                              -coords_gradients[i][0][1] * (coords_gradients[i][1][0] * coords_gradients[i][2][2] - coords_gradients[i][1][2] * coords_gradients[i][2][0])
                              +coords_gradients[i][0][2] * (coords_gradients[i][1][0] * coords_gradients[i][2][1] - coords_gradients[i][1][1] * coords_gradients[i][2][0]);
        }
    }
    return determinants;
}

// Integer root from
// https://rosettacode.org/wiki/Integer_roots#C.2B.2B
unsigned int root(unsigned int base, unsigned int n) {
    if (base < 2) return base;
    if (n == 0) return 1;

    unsigned int n1 = n - 1;
    unsigned int n2 = n;
    unsigned int n3 = n1;
    unsigned int c = 1;
    auto d = (n3 + base) / n2;
    auto e = (n3 * d + base / pow(d, n1)) / n2;

    while (c != d && c != e) {
        c = d;
        d = e;
        e = (n3*e + base / pow(e, n1)) / n2;
    }

    if (d < e) return d;
    return e;
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

    jacobian_determinants = determinant_ArrayTensor<dim,real>(quad_pts_coords_gradients);

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

template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_volume_term_explicit(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &/*metric_dof_indices*/,
    const unsigned int /*poly_degree*/,
    const unsigned int /*grid_degree*/,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    (void) current_cell_index;
    using doubleArray = std::array<real,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
    const unsigned int n_soln_dofs_int     = fe_values_vol.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());

    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();

    std::vector< doubleArray > soln_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts);

    std::vector< ADArrayTensor1 > conv_phys_flux_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
    std::vector< doubleArray > source_at_q(n_quad_pts);

    std::vector< real > soln_coeff(n_soln_dofs_int);
    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        soln_coeff[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices_int[idof]);
    }

    typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
        this->triangulation.get(), cell->level(), cell->index(), &(this->dof_handler_artificial_dissipation));
    const unsigned int n_dofs_arti_diss = this->fe_q_artificial_dissipation.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);
    artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);

    std::vector<real> artificial_diss_coeff_at_q(n_quad_pts);
    real max_artificial_diss = 0.0;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        artificial_diss_coeff_at_q[iquad] = 0.0;

        if ( this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ) {
            const dealii::Point<dim,real> point = fe_values_vol.get_quadrature().point(iquad);
            for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
                const unsigned int index = dof_indices_artificial_dissipation[idof];
                artificial_diss_coeff_at_q[iquad] += this->artificial_dissipation_c0[index] * this->fe_q_artificial_dissipation.shape_value(idof, point);
            }
            max_artificial_diss = std::max(artificial_diss_coeff_at_q[iquad], max_artificial_diss);
        }
    }

    const real artificial_diss_coeff = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
                                       this->artificial_dissipation_coeffs[current_cell_index]
                                       : 0.0;



    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) {
            // Interpolate solution to the face quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
              soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
        }
        // Evaluate physical convective flux and source term
        conv_phys_flux_at_q[iquad] = DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);
        if(this->all_parameters->artificial_dissipation_param.add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_phys_flux_at_q = DGBaseState<dim,nstate,real,MeshType>::artificial_dissip->calc_artificial_dissipation_flux (soln_at_q[iquad], soln_grad_at_q[iquad], artificial_diss_coeff);
            for (int istate=0; istate<nstate; istate++) {
                diss_phys_flux_at_q[iquad][istate] += artificial_diss_phys_flux_at_q[istate];
            }
        }
        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            const dealii::Point<dim,real> point = fe_values_vol.quadrature_point(iquad);
            source_at_q[iquad] = DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->source_term (point, soln_at_q[iquad], DGBase<dim,real,MeshType>::current_time);
            //std::array<real,nstate> artificial_source_at_q = DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->artificial_source_term (artificial_diss_coeff, point, soln_at_q[iquad]);
            //for (int s=0;s<nstate;++s) source_at_q[iquad][s] += artificial_source_at_q[s];
        }
    }

    const unsigned int cell_index = fe_values_vol.get_cell()->active_cell_index();
    const unsigned int cell_degree = fe_values_vol.get_fe().tensor_degree();
    real cell_volume = 0.0;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        cell_volume = cell_volume + JxW[iquad];
    }
    const real diameter = fe_values_vol.get_cell()->diameter();
    const real cell_diameter = cell_volume / std::pow(diameter,dim-1);
    //const real cell_diameter = std::pow(cell_volume,1.0/dim);
    //const real cell_diameter = cell_volume;
    const real cell_radius = 0.5 * cell_diameter;
    this->cell_volume[cell_index] = cell_volume;
    this->max_dt_cell[cell_index] = DGBaseState<dim,nstate,real,MeshType>::evaluate_CFL ( soln_at_q, max_artificial_diss, cell_radius, cell_degree);

    // Weak form
    // The right-hand side sends all the term to the side of the source term
    // Therefore,
    // \divergence ( Fconv + Fdiss ) = source
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {

        real rhs = 0;

        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // Convective
            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * conv_phys_flux_at_q[iquad][istate] * JxW[iquad];
            //// Diffusive
            //// Note that for diffusion, the negative is defined in the physics_double
            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW[iquad];
            // Source
            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
                rhs = rhs + fe_values_vol.shape_value_component(itest,iquad,istate) * source_at_q[iquad][istate] * JxW[iquad];
            }
        }

        local_rhs_int_cell(itest) += rhs;

    }
}


template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_boundary_term_explicit(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    (void) current_cell_index;
    using doubleArray = std::array<real,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    const unsigned int n_soln_dofs_int = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());

    const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim>> &normals = fe_values_boundary.get_normal_vectors ();


    std::vector<doubleArray> soln_int(n_face_quad_pts);
    std::vector<doubleArray> soln_ext(n_face_quad_pts);

    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);

    std::vector<doubleArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<doubleArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<doubleArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    // AD variable
    std::vector< real > soln_coeff_int(n_soln_dofs_int);
    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices_int[idof]);
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) {
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
        }
    }

    //const real cell_diameter = fe_values_boundary.get_cell()->diameter();
    //const real artificial_diss_coeff = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
    //                                   this->discontinuity_sensor(cell_diameter, soln_coeff_int, fe_values_boundary.get_fe())
    //                                   : 0.0;
    const real artificial_diss_coeff = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
                                       this->artificial_dissipation_coeffs[current_cell_index]
                                       : 0.0;

    //typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
    //    this->triangulation.get(), cell->level(), cell->index(), &(this->dof_handler_artificial_dissipation));
    //const unsigned int n_dofs_arti_diss = this->fe_q_artificial_dissipation.dofs_per_cell;
    //std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);
    //artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);

    //std::vector<real> artificial_diss_coeff_at_q(n_face_quad_pts);
    //for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
    //    artificial_diss_coeff_at_q[iquad] = 0.0;

    //    if ( this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ) {
    //        const dealii::Point<dim,real> point = fe_values_boundary.get_quadrature().point(iquad);
    //        for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
    //            const unsigned int index = dof_indices_artificial_dissipation[idof];
    //            artificial_diss_coeff_at_q[iquad] += this->artificial_dissipation_c0[index] * this->fe_q_artificial_dissipation.shape_value(idof, point);
    //        }
    //    }
    //}

    // Interpolate solution to face
    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,real> normal_int = normals[iquad];

        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
        }

        const dealii::Point<dim, real> real_quad_point = quad_pts[iquad];
        DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->boundary_face_values (boundary_id, real_quad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux_double->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            for (int d=0; d<dim; d++) {
                diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
            }
        }
        diss_flux_jump_int[iquad] = DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        if (this->all_parameters->artificial_dissipation_param.add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_flux_jump_int = DGBaseState<dim,nstate,real,MeshType>::artificial_dissip->calc_artificial_dissipation_flux(soln_int[iquad], diss_soln_jump_int,artificial_diss_coeff);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[iquad][s] += artificial_diss_flux_jump_int[s];
            }
        }

        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double->evaluate_auxiliary_flux(
            artificial_diss_coeff,
            artificial_diss_coeff,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    }

    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {

        real rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * conv_num_flux_dot_n[iquad][istate] * JxW[iquad];
            // Diffusive
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
            rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
        }
        // *******************

        local_rhs_int_cell(itest) += rhs;
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_face_term_explicit(
    const unsigned int /*iface*/, const unsigned int /*neighbor_iface*/,
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const unsigned int /*poly_degree*/, const unsigned int /*grid_degree*/,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &/*metric_dof_indices_int*/,
    const std::vector<dealii::types::global_dof_index> &/*metric_dof_indices_ext*/,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;
    using doubleArray = std::array<real,nstate>;
    using doubleArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_soln_dofs_int = fe_values_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_soln_dofs_ext, soln_dof_indices_ext.size());

    // Jacobian and normal should always be consistent between two elements
    // In the case of the non-conforming mesh, we should be using the Jacobian
    // of the smaller face since it would be "half" of the larger one.
    // This should be consistent with the DGBase decision of which cells is reponsible for
    // the face.
    // However, their curvature should match.
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();

    // AD variable
    std::vector<real> soln_coeff_int(n_soln_dofs_int);
    std::vector<real> soln_coeff_ext(n_soln_dofs_ext);

    std::vector<doubleArray> conv_num_flux_dot_n(n_face_quad_pts);

    // Interpolate solution to the face quadrature points
    std::vector< doubleArray > soln_int(n_face_quad_pts);
    std::vector< doubleArray > soln_ext(n_face_quad_pts);

    std::vector< doubleArrayTensor1 > soln_grad_int(n_face_quad_pts); // Tensor initialize with zeros
    std::vector< doubleArrayTensor1 > soln_grad_ext(n_face_quad_pts); // Tensor initialize with zeros

    std::vector<doubleArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<doubleArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<doubleArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<doubleArrayTensor1> diss_flux_jump_ext(n_face_quad_pts); // u*-u_ext
    // AD variable
    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices_int[idof]);
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
        soln_coeff_ext[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices_ext[idof]);
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) {
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
            soln_ext[iquad][istate]      = 0;
            soln_grad_ext[iquad][istate] = 0;
        }
    }

    //const real cell_diameter_int = fe_values_int.get_cell()->diameter();
    //const real cell_diameter_ext = fe_values_ext.get_cell()->diameter();
    //const real artificial_diss_coeff_int = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
    //                                       this->discontinuity_sensor(cell_diameter_int, soln_coeff_int, fe_values_int.get_fe())
    //                                       : 0.0;
    //const real artificial_diss_coeff_ext = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
    //                                       this->discontinuity_sensor(cell_diameter_ext, soln_coeff_ext, fe_values_ext.get_fe())
    //                                       : 0.0;
    const real artificial_diss_coeff_int = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
                                           this->artificial_dissipation_coeffs[current_cell_index]
                                           : 0.0;
    const real artificial_diss_coeff_ext = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
                                           this->artificial_dissipation_coeffs[neighbor_cell_index]
                                           : 0.0;

    //typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
    //    this->triangulation.get(), cell->level(), cell->index(), &(this->dof_handler_artificial_dissipation));
    //const unsigned int n_dofs_arti_diss = this->fe_q_artificial_dissipation.dofs_per_cell;
    //std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);
    //artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);

    //std::vector<real> artificial_diss_coeff_at_q(n_face_quad_pts);
    //for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
    //    artificial_diss_coeff_at_q[iquad] = 0.0;

    //    if ( this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ) {
    //        const dealii::Point<dim,real> point = fe_values_int.get_quadrature().point(iquad);
    //        for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
    //            const unsigned int index = dof_indices_artificial_dissipation[idof];
    //            artificial_diss_coeff_at_q[iquad] += this->artificial_dissipation_c0[index] * this->fe_q_artificial_dissipation.shape_value(idof, point);
    //        }
    //    }
    //}

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,real> normal_int = normals_int[iquad];
        const dealii::Tensor<1,dim,real> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
            const unsigned int istate = fe_values_int.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_int.shape_grad_component(idof, iquad, istate);
        }
        for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
            const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
            soln_ext[iquad][istate]      += soln_coeff_ext[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
            soln_grad_ext[iquad][istate] += soln_coeff_ext[idof] * fe_values_ext.shape_grad_component(idof, iquad, istate);
        }

        // Evaluate physical convective flux, physical dissipative flux, and source term
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        doubleArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            for (int d=0; d<dim; d++) {
                diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
                diss_soln_jump_ext[s][d] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext[d];
            }
        }
        diss_flux_jump_int[iquad] = DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

        if (this->all_parameters->artificial_dissipation_param.add_artificial_dissipation) {
            const doubleArrayTensor1 artificial_diss_flux_jump_int = DGBaseState<dim,nstate,real,MeshType>::artificial_dissip->calc_artificial_dissipation_flux(soln_int[iquad], diss_soln_jump_int, artificial_diss_coeff_int);
            const doubleArrayTensor1 artificial_diss_flux_jump_ext = DGBaseState<dim,nstate,real,MeshType>::artificial_dissip->calc_artificial_dissipation_flux(soln_ext[iquad], diss_soln_jump_ext, artificial_diss_coeff_ext);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[iquad][s] += artificial_diss_flux_jump_int[s];
                diss_flux_jump_ext[iquad][s] += artificial_diss_flux_jump_ext[s];
            }
        }

        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double->evaluate_auxiliary_flux(
            artificial_diss_coeff_int,
            artificial_diss_coeff_ext,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty);
    }

    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
        real rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * conv_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
            rhs = rhs + fe_values_int.shape_grad_component(itest_int,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW_int[iquad];
        }

        local_rhs_int_cell(itest_int) += rhs;
    }

    // From test functions associated with neighbor cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
        real rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-conv_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-diss_auxi_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
            rhs = rhs + fe_values_ext.shape_grad_component(itest_ext,iquad,istate) * diss_flux_jump_ext[iquad][istate] * JxW_int[iquad];
        }

        local_rhs_ext_cell(itest_ext) += rhs;
    }
}

template <int dim, int nstate, typename real2>
void compute_br2_correction(
    const dealii::FESystem<dim,dim> &fe_soln,
    const std::vector< real2 > &coords_coeff,
    const dealii::FESystem<dim,dim> &fe_metric,
    const std::vector< std::array<real2,nstate> > &lifting_op_R_rhs,
    std::vector< std::array<real2,nstate> > &soln_grad_correction
    )
{
    const unsigned int n_faces = std::pow(2,dim);
    const double br2_factor = n_faces * 1.01;

    // Get the base finite element
    // Assumption is that the vector-valued finite element uses the same basis for every state equation.
    const dealii::FiniteElement<dim> &base_fe = fe_soln.get_sub_fe(0,1);
    const unsigned int n_base_dofs = base_fe.n_dofs_per_cell();

    // Build lifting term of BR2
    // For this purposes of BR2, do NOT overintegrate to have a square invertible differentiation matrix
    const int degree = base_fe.tensor_degree();
    dealii::QGauss<dim> vol_quad(degree+1);
    const unsigned int n_vol_quad = vol_quad.size();

    if (n_base_dofs != n_vol_quad) std::abort();

    // Obtain metric Jacobians at volume quadratures.
    const std::vector<dealii::Point<dim,double>> &vol_unit_quad_pts = vol_quad.get_points();
    using Tensor2D = dealii::Tensor<2,dim,real2>;
    std::vector<Tensor2D> volume_metric_jac = evaluate_metric_jacobian (vol_unit_quad_pts, coords_coeff, fe_metric);

    // Evaluate Vandermonde operator
    dealii::FullMatrix<double> vandermonde_inverse(n_base_dofs, n_vol_quad);

    for (unsigned int idof_base=0; idof_base<n_base_dofs; ++idof_base) {
        for (unsigned int iquad=0; iquad<n_vol_quad; ++iquad) {
            vandermonde_inverse[idof_base][iquad] = base_fe.shape_value(idof_base, vol_quad.point(iquad));
        }
    }
    gauss_jordan(vandermonde_inverse);

    // Print Inverse Mass matrix
    //{
    //dealii::FullMatrix<real2> vandermonde_inverse_real(n_base_dofs, n_vol_quad);

    //for (unsigned int idof_base=0; idof_base<n_base_dofs; ++idof_base) {
    //    for (unsigned int iquad=0; iquad<n_vol_quad; ++iquad) {
    //        vandermonde_inverse_real[idof_base][iquad] = base_fe.shape_value(idof_base, vol_quad.point(iquad));
    //    }
    //}
    //gauss_jordan(vandermonde_inverse);
    //dealii::FullMatrix<real2> diag_jac(n_base_dofs, n_base_dofs);
    //diag_jac = 0.0;
    //for (unsigned int kquad=0; kquad<n_vol_quad; ++kquad) {
    //    diag_jac[kquad][kquad] = 1.0/(dealii::determinant(volume_metric_jac[kquad]) * vol_quad.weight(kquad));
    //}

    //dealii::FullMatrix<real2> lifting_operator1(n_base_dofs, n_base_dofs);
    //diag_jac.mmult(lifting_operator1, vandermonde_inverse_real);

    //dealii::FullMatrix<real2> lifting_operator2(n_base_dofs, n_base_dofs);
    //vandermonde_inverse_real.Tmmult(lifting_operator2, lifting_operator1);

    //std::cout << "Inverse mass..." << std::endl;
    //lifting_operator2.print(std::cout);
    //}

    std::vector< std::array<real2,nstate> > vandermonde_inv_rhs(n_vol_quad);
    for (unsigned int kquad=0; kquad<n_vol_quad; ++kquad) {
        for (int s=0; s<nstate; s++) {
            vandermonde_inv_rhs[kquad][s] = 0.0;
            for (unsigned int jdof_base=0; jdof_base<n_base_dofs; ++jdof_base) {
                vandermonde_inv_rhs[kquad][s] += vandermonde_inverse[kquad][jdof_base] * lifting_op_R_rhs[jdof_base][s];
            }
        }
    }
    for (unsigned int kquad=0; kquad<n_vol_quad; ++kquad) {
        for (int s=0; s<nstate; s++) {
            vandermonde_inv_rhs[kquad][s] /= dealii::determinant(volume_metric_jac[kquad]) * vol_quad.weight(kquad);
        }
    }
    for (unsigned int idof_base=0; idof_base<n_base_dofs; ++idof_base) {
        for (int s=0; s<nstate; s++) {
            soln_grad_correction[idof_base][s] = 0.0;
            for (unsigned int kquad=0; kquad<n_vol_quad; ++kquad) {
                soln_grad_correction[idof_base][s] += vandermonde_inverse[kquad][idof_base] * vandermonde_inv_rhs[kquad][s];
            }
            soln_grad_correction[idof_base][s] *= br2_factor;
            //soln_grad_correction[idof_base][s] /= dim; // Due to the dot-product of the vector-valued mass matrix
        }
    }
}

template <int dim, int nstate, typename real2>
void correct_the_gradient(
    const std::vector<std::array<real2,nstate>>                          &soln_grad_corr,
    const dealii::FESystem<dim,dim>                                      &fe_soln,
    const std::vector<std::array<dealii::Tensor<1,dim,real2>, nstate>>   &soln_jump,
    const dealii::FullMatrix<double>                                     &interpolation_operator,
    const std::array<dealii::FullMatrix<real2>,dim>                      &gradient_operator,
    std::vector<std::array< dealii::Tensor<1,dim,real2>, nstate >>       &soln_grad)
{
    (void) soln_jump;
    (void) soln_grad_corr;
    (void) interpolation_operator;
    (void) gradient_operator;
    const unsigned int n_quad = soln_grad.size();
    const unsigned int n_soln_dofs = fe_soln.dofs_per_cell;

    for (unsigned int iquad=0; iquad<n_quad; ++iquad) {
        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
            const unsigned int istate = fe_soln.system_to_component_index(idof).first;
            const unsigned int idof_base = fe_soln.system_to_component_index(idof).second;
            (void) istate;
            (void) idof_base;
            for (int d=0;d<dim;++d) {
                //soln_grad[iquad][istate][d] += soln_jump[iquad][istate][d];
                soln_grad[iquad][istate][d] += soln_grad_corr[idof_base][istate] * interpolation_operator[idof][iquad];
                //soln_grad[iquad][istate][d] += soln_grad_corr[idof_base][istate] * gradient_operator[d][idof][iquad];
            }
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
void DGWeak<dim,nstate,real,MeshType>::assemble_boundary_term(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const std::vector< real2 > &soln_coeff,
    const std::vector< real2 > &coords_coeff,
    const std::vector< real > &local_dual,
    const unsigned int face_number,
    const unsigned int boundary_id,
    const Physics::PhysicsBase<dim, nstate, real2> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, real2> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, real2> &diss_num_flux,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::FESystem<dim,dim> &fe_metric,
    const dealii::Quadrature<dim-1> &quadrature,
    std::vector<real2> &rhs,
    real2 &dual_dot_residual,
    const bool compute_metric_derivatives)
{
    const unsigned int n_soln_dofs = fe_soln.dofs_per_cell;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_quad_pts = fe_values_boundary.n_quadrature_points;

    dual_dot_residual = 0.0;
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        rhs[itest] = 0.0;
    }

    using ADArray = std::array<real2,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,real2>, nstate >;
    using Tensor2D = dealii::Tensor<2,dim,real2>;

    const dealii::Quadrature<dim> face_quadrature
        = dealii::QProjector<dim>::project_to_face(
            dealii::ReferenceCell::get_hypercube(dim),
            quadrature,
            face_number);
    const std::vector<dealii::Point<dim,real>> &unit_quad_pts = face_quadrature.get_points();
    std::vector<dealii::Point<dim,real2>> real_quad_pts(unit_quad_pts.size());

    std::vector<dealii::Tensor<2,dim,real2>> metric_jacobian = evaluate_metric_jacobian (unit_quad_pts, coords_coeff, fe_metric);
    std::vector<real2> jac_det(n_quad_pts);
    std::vector<real2> surface_jac_det(n_quad_pts);
    std::vector<dealii::Tensor<2,dim,real2>> jac_inv_tran(n_quad_pts);

    const dealii::Tensor<1,dim,real> unit_normal = dealii::GeometryInfo<dim>::unit_normal_vector[face_number];
    std::vector<dealii::Tensor<1,dim,real2>> phys_unit_normal(n_quad_pts);

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        if (compute_metric_derivatives) {
            for (int d=0;d<dim;++d) { real_quad_pts[iquad][d] = 0;}
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const int iaxis = fe_metric.system_to_component_index(idof).first;
                real_quad_pts[iquad][iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
            }

            const real2 jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
            const dealii::Tensor<2,dim,real2> jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));

            jac_det[iquad] = jacobian_determinant;
            jac_inv_tran[iquad] = jacobian_transpose_inverse;

            const dealii::Tensor<1,dim,real2> normal = vmult(jacobian_transpose_inverse, unit_normal);
            const real2 area = norm(normal);

            surface_jac_det[iquad] = norm(normal)*jac_det[iquad];
            // Technically the normals have jac_det multiplied.
            // However, we use normalized normals by convention, so the the term
            // ends up appearing in the surface jacobian.
            for (int d=0;d<dim;++d) { 
                phys_unit_normal[iquad][d] = normal[d] / area;
            }

            // Exact mapping
            // real_quad_pts[iquad] = fe_values_boundary.quadrature_point(iquad);
            // surface_jac_det[iquad] = fe_values_boundary.JxW(iquad) / face_quadrature.weight(iquad);
            // phys_unit_normal[iquad] = fe_values_boundary.normal_vector(iquad);

        } else {
            real_quad_pts[iquad] = fe_values_boundary.quadrature_point(iquad);
            surface_jac_det[iquad] = fe_values_boundary.JxW(iquad) / face_quadrature.weight(iquad);
            phys_unit_normal[iquad] = fe_values_boundary.normal_vector(iquad);
        }

    }
#ifdef KOPRIVA_METRICS_BOUNDARY
    auto old_jac_det = jac_det;
    auto old_jac_inv_tran = jac_inv_tran;

    if constexpr (dim != 1) {
        evaluate_covariant_metric_jacobian<dim,real2> ( face_quadrature, coords_coeff, fe_metric, jac_inv_tran, jac_det);
    }

    //for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //    //if (abs(old_jac_det[iquad] - jac_det[iquad]) > 1e-10) {
    //        std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    //        std::cout << "Not the same jac det int, iquad " << iquad << std::endl;
    //        std::cout << old_jac_det[iquad] << std::endl;
    //        std::cout << jac_det[iquad] << std::endl;
    //    //}
    //    for (int row=0;row<dim;++row) {
    //        for (int col=0;col<dim;++col) {
    //            //if (abs(old_jac_inv_tran[iquad][row][col] - jac_inv_tran[iquad][row][col]) > 1e-10) {
    //                std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    //                std::cout << "Not the same jac inv tran int, iquad " << iquad << " row " << row << " col " << col << std::endl;
    //                std::cout << old_jac_inv_tran[iquad][row][col] << std::endl;
    //                std::cout << jac_inv_tran[iquad][row][col] << std::endl;
    //            }
    //        //}
    //    }
    //}
#endif

    std::vector<real2> faceJxW(n_quad_pts);

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        if (compute_metric_derivatives) {
            const dealii::Tensor<1,dim,real2> normal = vmult(jac_inv_tran[iquad], unit_normal);
            const real2 area = norm(normal);

            surface_jac_det[iquad] = norm(normal)*jac_det[iquad];
            // Technically the normals have jac_det multiplied.
            // However, we use normalized normals by convention, so the the term
            // ends up appearing in the surface jacobian.
            for (int d=0;d<dim;++d) { 
                phys_unit_normal[iquad][d] = normal[d] / area;
            }
        }

        faceJxW[iquad] = surface_jac_det[iquad] * face_quadrature.weight(iquad);
    }

    dealii::FullMatrix<real> interpolation_operator(n_soln_dofs,n_quad_pts);
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            interpolation_operator[idof][iquad] = fe_soln.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    std::array<dealii::FullMatrix<real2>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(dealii::TableIndices<2>(n_soln_dofs, n_quad_pts));
    }
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            if (compute_metric_derivatives) {
                const dealii::Tensor<1,dim,real> ref_shape_grad = fe_soln.shape_grad(idof,unit_quad_pts[iquad]);
                const dealii::Tensor<1,dim,real2> phys_shape_grad = vmult(jac_inv_tran[iquad], ref_shape_grad);
                for (int d=0;d<dim;++d) {
                    gradient_operator[d][idof][iquad] = phys_shape_grad[d];
                }

                // Exact mapping
                // for (int d=0;d<dim;++d) {
                //     const unsigned int istate = fe_soln.system_to_component_index(idof).first;
                //     gradient_operator[d][idof][iquad] = fe_values_boundary.shape_grad_component(idof, iquad, istate)[d];
                // }
            } else {
                for (int d=0;d<dim;++d) {
                    const unsigned int istate = fe_soln.system_to_component_index(idof).first;
                    gradient_operator[d][idof][iquad] = fe_values_boundary.shape_grad_component(idof, iquad, istate)[d];
                }
            }
        }
    }


    std::vector<std::array<real2,nstate>> soln_int(n_quad_pts), soln_ext(n_quad_pts);
    std::vector<std::array< dealii::Tensor<1,dim,real2>, nstate >> soln_grad_int(n_quad_pts), soln_grad_ext(n_quad_pts);

    evaluate_finite_element_values<dim, real2, nstate> (unit_quad_pts, soln_coeff, fe_soln, soln_int);

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        for (int istate=0; istate<nstate; istate++) {
            soln_grad_int[iquad][istate] = 0;
        }
        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            for (int d=0;d<dim;++d) {
                soln_grad_int[iquad][istate][d] += soln_coeff[idof] * gradient_operator[d][idof][iquad];
            }
        }
    }

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        const dealii::Tensor<1,dim,real2> normal_int = phys_unit_normal[iquad];
        physics.boundary_face_values (boundary_id, real_quad_pts[iquad], normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);
    }

    // Assemble BR2 gradient correction right-hand side
    const dealii::FiniteElement<dim> &base_fe_int = fe_soln.get_sub_fe(0,1);
    const unsigned int n_base_dofs_int = base_fe_int.n_dofs_per_cell();

    std::vector<ADArrayTensor1> soln_grad_correction_int(n_base_dofs_int);
    using DissFlux = Parameters::AllParameters::DissipativeNumericalFlux;
    if (this->all_parameters->diss_num_flux_type == DissFlux::bassi_rebay_2) {

        // Obtain solution jump
        std::vector<std::array<dealii::Tensor<1,dim,real2>, nstate>> soln_jump_int(n_quad_pts);
        std::vector<std::array<dealii::Tensor<1,dim,real2>, nstate>> soln_jump_ext(n_quad_pts);
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (int s=0; s<nstate; s++) {
                for (int d=0; d<dim; d++) {
                    soln_jump_int[iquad][s][d] = (soln_int[iquad][s] - soln_ext[iquad][s]) * (phys_unit_normal[iquad][d]);
                    soln_jump_ext[iquad][s][d] = (soln_ext[iquad][s] - soln_int[iquad][s]) * (-phys_unit_normal[iquad][d]);
                }
            }
        }

        std::vector<ADArray> lifting_op_R_rhs_int(n_base_dofs_int);
        for (unsigned int idof_base=0; idof_base<n_base_dofs_int; ++idof_base) {
            for (int s=0; s<nstate; s++) {

                const unsigned int idof = fe_soln.component_to_system_index(s, idof_base);
                lifting_op_R_rhs_int[idof_base][s] = 0.0;

                for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                    for (int d=0; d<dim; ++d) {
                        //const real2 basis_average = gradient_operator[d][idof][iquad];
                        const double basis_average = interpolation_operator[idof][iquad];
                        lifting_op_R_rhs_int[idof_base][s] -= soln_jump_int[iquad][s][d] * basis_average * faceJxW[iquad];
                    }
                }

            }
        }
        std::vector<ADArray> soln_grad_corr_int(n_base_dofs_int);
        compute_br2_correction<dim,nstate,real2>(fe_soln, coords_coeff, fe_metric, lifting_op_R_rhs_int, soln_grad_corr_int);

        correct_the_gradient<dim,nstate,real2>( soln_grad_corr_int, fe_soln, soln_jump_int, interpolation_operator, gradient_operator, soln_grad_int);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
                const unsigned int istate = fe_soln.system_to_component_index(idof).first;
                const unsigned int idof_base = fe_soln.system_to_component_index(idof).second;
                (void) istate;
                (void) idof_base;
                for (int d=0;d<dim;++d) {
                    //soln_grad_int[iquad][istate][d] += soln_jump_int[iquad][istate][d];
                    //soln_grad_int[iquad][istate][d] += soln_grad_corr_int[idof_base][istate] * interpolation_operator[idof][iquad];
                    //soln_grad_int[iquad][istate][d] += soln_grad_corr_int[idof_base][istate] * gradient_operator[d][idof][iquad];
                    //soln_grad_ext[iquad][istate][d] -= soln_grad_corr_int[idof_base][istate] * gradient_operator[d][idof][iquad];
                    soln_grad_ext[iquad][istate][d] = soln_grad_int[iquad][istate][d];
                }
            }
            physics.boundary_face_values (boundary_id, real_quad_pts[iquad], phys_unit_normal[iquad], soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);
        }

    } 


    std::vector<ADArray> conv_num_flux_dot_n(n_quad_pts);
    std::vector<ADArray> diss_soln_num_flux(n_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_quad_pts); // u*-u_int
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_quad_pts); // sigma*

    //const real2 cell_diameter = fe_values_boundary.get_cell()->diameter();
    //const real2 artificial_diss_coeff = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
    //                                       this->discontinuity_sensor(cell_diameter, soln_coeff, fe_values_boundary.get_fe())
    //                                       : 0.0;
    const real2 artificial_diss_coeff = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
                                         this->artificial_dissipation_coeffs[current_cell_index]
                                         : 0.0;
    (void) artificial_diss_coeff;

    typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
        this->triangulation.get(), cell->level(), cell->index(), &(this->dof_handler_artificial_dissipation));
    const unsigned int n_dofs_arti_diss = this->fe_q_artificial_dissipation.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);
    artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);

    std::vector<real> artificial_diss_coeff_at_q(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        artificial_diss_coeff_at_q[iquad] = 0.0;
        if ( this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ) {
            const dealii::Point<dim,real> point = unit_quad_pts[iquad];
            for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
                const unsigned int index = dof_indices_artificial_dissipation[idof];
                artificial_diss_coeff_at_q[iquad] += this->artificial_dissipation_c0[index] * this->fe_q_artificial_dissipation.shape_value(idof, point);
            }
        }
        artificial_diss_coeff_at_q[iquad] = 0.0;
    }

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,real2> normal_int = phys_unit_normal[iquad];

        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux_fad_fad->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        //conv_num_flux_dot_n[iquad] = conv_num_flux.evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        conv_num_flux_dot_n[iquad] = conv_num_flux.evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = diss_num_flux.evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            for (int d=0; d<dim; d++) {
                diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
            }
        }
        diss_flux_jump_int[iquad] = physics.dissipative_flux (soln_int[iquad], diss_soln_jump_int);

        if (this->all_parameters->artificial_dissipation_param.add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_flux_jump_int = DGBaseState<dim,nstate,real,MeshType>::artificial_dissip->calc_artificial_dissipation_flux(soln_int[iquad], diss_soln_jump_int,artificial_diss_coeff_at_q[iquad]);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[iquad][s] += artificial_diss_flux_jump_int[s];
            }
        }

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux.evaluate_auxiliary_flux(
            //artificial_diss_coeff,
            //artificial_diss_coeff,
            artificial_diss_coeff_at_q[iquad],
            artificial_diss_coeff_at_q[iquad],
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    }

    // Applying convection boundary condition
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

        real2 rhs_val = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const real2 JxW_iquad = faceJxW[iquad];
            // Convection
            rhs_val = rhs_val - interpolation_operator[itest][iquad] * conv_num_flux_dot_n[iquad][istate] * JxW_iquad;
            // Diffusive
            rhs_val = rhs_val - interpolation_operator[itest][iquad] * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs_val = rhs_val + gradient_operator[d][itest][iquad] * diss_flux_jump_int[iquad][istate][d] * JxW_iquad;
            }
        }


        rhs[itest] = rhs_val;
        dual_dot_residual += local_dual[itest]*rhs_val;
    }
}

#ifdef FADFAD
template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_boundary_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;
    using adtype = FadFadType;

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_soln_dofs = fe_values_boundary.dofs_per_cell;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    const bool compute_metric_derivatives = true;//(!compute_dRdX && !compute_d2R) ? false : true;
    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    std::vector< adtype > soln_coeff(n_soln_dofs);
    std::vector< adtype > coords_coeff(n_metric_dofs);

    unsigned int w_start, w_end, x_start, x_end;
    automatic_differentiation_indexing_1( compute_dRdW, compute_dRdX, compute_d2R,
                                          n_soln_dofs, n_metric_dofs,
                                          w_start, w_end, x_start, x_end );

    unsigned int i_derivative = 0;
    const unsigned int n_total_indep = x_end;

    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;
        soln_coeff[idof].val() = val;

        if (compute_dRdW || compute_d2R) soln_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) soln_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdW || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;
        coords_coeff[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdX || compute_d2R) i_derivative++;
    }

    AssertDimension(i_derivative, n_total_indep);

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_dual[itest] = this->dual[soln_dof_indices[itest]];
    }

    const auto &physics = *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_fad_fad);
    const auto &conv_num_flux = *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad_fad);
    const auto &diss_num_flux = *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad_fad);

    std::vector<adtype> rhs(n_soln_dofs);
    adtype dual_dot_residual;
    assemble_boundary_term(
        cell,
        current_cell_index,
        soln_coeff,
        coords_coeff,
        local_dual,
        face_number,
        boundary_id,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_boundary,
        penalty,
        fe_soln,
        fe_metric,
        quadrature,
        rhs,
        dual_dot_residual,
        compute_metric_derivatives);

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_rhs_cell[itest] += rhs[itest].val().val();
    }

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        if (compute_dRdW) {
            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = rhs[itest].dx(i_dx).val();
                AssertIsFinite(residual_derivatives[idof]);
            }
            const bool elide_zero_values = false;
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives, elide_zero_values);
        }
        if (compute_dRdX) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = rhs[itest].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }

    }

    if (compute_d2R) {
        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);
        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;
            const FadType dWi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_soln_dofs; ++jdof) {
                const unsigned int j_dx = jdof+w_start;
                dWidW[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices[idof], soln_dof_indices, dWidW);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices[idof], metric_dof_indices, dWidX);
        }
        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_start;
            const FadType dXi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = dXi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }
    }
}
#endif

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGWeak<dim,nstate,real,MeshType>::assemble_boundary_codi_taped_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    const Physics::PhysicsBase<dim, nstate, adtype> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
    dealii::Vector<real> &local_rhs_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_soln_dofs = fe_values_boundary.dofs_per_cell;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    const bool compute_metric_derivatives = true;//(!compute_dRdX && !compute_d2R) ? false : true;
    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    std::vector< adtype > soln_coeff(n_soln_dofs);
    std::vector< adtype > coords_coeff(n_metric_dofs);

    unsigned int w_start, w_end, x_start, x_end;
    automatic_differentiation_indexing_1( compute_dRdW, compute_dRdX, compute_d2R,
                                          n_soln_dofs, n_metric_dofs,
                                          w_start, w_end, x_start, x_end );

    using TH = codi::TapeHelper<adtype>;
    TH th;
    adtype::getGlobalTape();
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.startRecording();
    }
    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;

        if (compute_dRdW || compute_d2R) {
            th.registerInput(soln_coeff[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(soln_coeff[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;

        if (compute_dRdX || compute_d2R) {
            th.registerInput(coords_coeff[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(coords_coeff[idof]);
        }
    }

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_dual[itest] = this->dual[soln_dof_indices[itest]];
    }

    std::vector<adtype> rhs(n_soln_dofs);
    adtype dual_dot_residual;
    assemble_boundary_term(
        cell,
        current_cell_index,
        soln_coeff,
        coords_coeff,
        local_dual,
        face_number,
        boundary_id,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_boundary,
        penalty,
        fe_soln,
        fe_metric,
        quadrature,
        rhs,
        dual_dot_residual,
        compute_metric_derivatives);

    if (compute_dRdW || compute_dRdX) {
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
            th.registerOutput(rhs[itest]);
        }
    } else if (compute_d2R) {
        th.registerOutput(dual_dot_residual);
    }
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.stopRecording();
    }

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_rhs_cell(itest) += getValue<adtype>(rhs[itest]);
        AssertIsFinite(local_rhs_cell(itest));
    }

    if (compute_dRdW) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = jac(itest,i_dx);
                AssertIsFinite(residual_derivatives[idof]);
            }
            const bool elide_zero_values = false;
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives, elide_zero_values);
        }
        th.deleteJacobian(jac);

    }

    if (compute_dRdX) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = jac(itest,i_dx);
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }
        th.deleteJacobian(jac);
    }


    if (compute_d2R) {
        typename TH::HessianType& hes = th.createHessian();
        th.evalHessian(hes);

        int i_dependent = (compute_dRdW || compute_dRdX) ? n_soln_dofs : 0;

        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);

        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;

            for (unsigned int jdof=0; jdof<n_soln_dofs; ++jdof) {
                const unsigned int j_dx = jdof+w_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices[idof], soln_dof_indices, dWidW);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices[idof], metric_dof_indices, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_start;

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }

        th.deleteHessian(hes);
    }
    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        adtype::getGlobalTape().deactivateValue(soln_coeff[idof]);
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        adtype::getGlobalTape().deactivateValue(coords_coeff[idof]);
    }

}

template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_boundary_residual(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    const Physics::PhysicsBase<dim, nstate, real> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, real> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, real> &diss_num_flux,
    dealii::Vector<real> &local_rhs_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_soln_dofs = fe_values_boundary.dofs_per_cell;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    (void) compute_dRdW;
    (void) compute_dRdX;
    (void) compute_d2R;

    const bool compute_metric_derivatives = true; //= (!compute_dRdX && !compute_d2R) ? false : true;
    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    std::vector< real > soln_coeff(n_soln_dofs);
    std::vector< real > coords_coeff(n_metric_dofs);

    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;
    }

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_dual[itest] = this->dual[soln_dof_indices[itest]];
    }

    std::vector<real> rhs(n_soln_dofs);
    real dual_dot_residual;
    assemble_boundary_term(
        cell,
        current_cell_index,
        soln_coeff,
        coords_coeff,
        local_dual,
        face_number,
        boundary_id,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_boundary,
        penalty,
        fe_soln,
        fe_metric,
        quadrature,
        rhs,
        dual_dot_residual,
        compute_metric_derivatives);

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_rhs_cell(itest) += getValue<real>(rhs[itest]);
        AssertIsFinite(local_rhs_cell(itest));
    }

}

#ifndef FADFAD
template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_boundary_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;
    if (compute_d2R) {
        assemble_boundary_codi_taped_derivatives<codi_HessianComputationType>(
        cell,
            current_cell_index,
            face_number,
            boundary_id,
            fe_values_boundary,
            penalty,
            fe_soln,
            quadrature,
            metric_dof_indices,
            soln_dof_indices,
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad_fad),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_rad_fad),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_rad_fad),
            local_rhs_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    } else if (compute_dRdW || compute_dRdX) {
        assemble_boundary_codi_taped_derivatives<codi_JacobianComputationType>(
        cell,
            current_cell_index,
            face_number,
            boundary_id,
            fe_values_boundary,
            penalty,
            fe_soln,
            quadrature,
            metric_dof_indices,
            soln_dof_indices,
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_rad),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_rad),
            local_rhs_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    } else {
        assemble_boundary_residual(
        cell,
            current_cell_index,
            face_number,
            boundary_id,
            fe_values_boundary,
            penalty,
            fe_soln,
            quadrature,
            metric_dof_indices,
            soln_dof_indices,
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_double),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_double),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double),
            local_rhs_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    }
}
#endif


template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
void DGWeak<dim,nstate,real,MeshType>::assemble_face_term(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const std::vector< real2 > &soln_coeff_int,
    const std::vector< real2 > &soln_coeff_ext,
    const std::vector< real2 > &coords_coeff_int,
    const std::vector< real2 > &coords_coeff_ext,
    const std::vector< double > &dual_int,
    const std::vector< double > &dual_ext,
    const std::pair<unsigned int, int> face_subface_int,
    const std::pair<unsigned int, int> face_subface_ext,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
    const Physics::PhysicsBase<dim, nstate, real2> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, real2> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, real2> &diss_num_flux,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_int,
    const dealii::FESystem<dim,dim> &fe_ext,
    const dealii::FESystem<dim,dim> &fe_metric,
    const dealii::Quadrature<dim-1> &face_quadrature,
    std::vector<real2> &rhs_int,
    std::vector<real2> &rhs_ext,
    real2 &dual_dot_residual,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) compute_dRdW;
    const unsigned int n_soln_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_ext.dofs_per_cell;
    const unsigned int n_face_quad_pts = face_quadrature.size();

    dual_dot_residual = 0.0;
    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {
        rhs_int[itest] = 0.0;
    }
    for (unsigned int itest=0; itest<n_soln_dofs_ext; ++itest) {
        rhs_ext[itest] = 0.0;
    }

    using ADArray = std::array<real2,nstate>;
    using Tensor1D = dealii::Tensor<1,dim,real2>;
    using Tensor2D = dealii::Tensor<2,dim,real2>;
    using ADArrayTensor1 = std::array< Tensor1D, nstate >;

    (void) face_data_set_int; (void) face_data_set_ext;
    dealii::Quadrature<dim> face_quadrature_int, face_quadrature_ext;
    if constexpr (dim < 3) {
        face_quadrature_int = face_subface_int.second == -1 ?
                              dealii::QProjector<dim>::project_to_face(dealii::ReferenceCell::get_hypercube(dim),
                                                                       face_quadrature,
                                                                       face_subface_int.first):
                              dealii::QProjector<dim>::project_to_subface(dealii::ReferenceCell::get_hypercube(dim),
                                                                       face_quadrature,
                                                                       face_subface_int.first,
                                                                       face_subface_int.second,
                                                                       dealii::RefinementCase<dim-1>::isotropic_refinement);
        face_quadrature_ext = face_subface_ext.second == -1 ?
                              dealii::QProjector<dim>::project_to_face(dealii::ReferenceCell::get_hypercube(dim),
                                                                       face_quadrature,
                                                                       face_subface_ext.first):
                              dealii::QProjector<dim>::project_to_subface(dealii::ReferenceCell::get_hypercube(dim),
                                                                       face_quadrature,
                                                                       face_subface_ext.first,
                                                                       face_subface_ext.second,
                                                                       dealii::RefinementCase<dim-1>::isotropic_refinement);
    } else {
        const dealii::Quadrature<dim> all_faces_quad_int = face_subface_int.second == -1 ?
                                                           dealii::QProjector<dim>::project_to_all_faces (dealii::ReferenceCell::get_hypercube(dim), face_quadrature) :
                                                           dealii::QProjector<dim>::project_to_all_subfaces (dealii::ReferenceCell::get_hypercube(dim), face_quadrature);
        const dealii::Quadrature<dim> all_faces_quad_ext = face_subface_ext.second == -1 ?
                                                           dealii::QProjector<dim>::project_to_all_faces (dealii::ReferenceCell::get_hypercube(dim), face_quadrature) :
                                                           dealii::QProjector<dim>::project_to_all_subfaces (dealii::ReferenceCell::get_hypercube(dim), face_quadrature);
        std::vector< dealii::Point< dim >> points(n_face_quad_pts);
        std::vector< double > weights(n_face_quad_pts);
        for (unsigned int iquad = 0; iquad < n_face_quad_pts; ++iquad) {
            points[iquad] = all_faces_quad_int.point(iquad+face_data_set_int);
            weights[iquad] = all_faces_quad_int.weight(iquad+face_data_set_int);
        }
        face_quadrature_int = dealii::Quadrature<dim>(points, weights);

        for (unsigned int iquad = 0; iquad < n_face_quad_pts; ++iquad) {
            points[iquad] = all_faces_quad_ext.point(iquad+face_data_set_ext);
            weights[iquad] = all_faces_quad_ext.weight(iquad+face_data_set_ext);
        }
        face_quadrature_ext = dealii::Quadrature<dim>(points, weights);
    }


    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    const bool compute_metric_derivatives = true; //(!compute_dRdX && !compute_d2R) ? false : true;

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_int = face_quadrature_int.get_points();
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_ext = face_quadrature_ext.get_points();



    // Use the metric Jacobian from the interior cell
    std::vector<Tensor2D> metric_jac_int = evaluate_metric_jacobian (unit_quad_pts_int, coords_coeff_int, fe_metric);
    std::vector<Tensor2D> metric_jac_ext = evaluate_metric_jacobian (unit_quad_pts_ext, coords_coeff_ext, fe_metric);
    std::vector<Tensor2D> jac_inv_tran_int(n_face_quad_pts);
    std::vector<Tensor2D> jac_inv_tran_ext(n_face_quad_pts);

    const dealii::Tensor<1,dim,real> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[face_subface_int.first];
    const dealii::Tensor<1,dim,real> unit_normal_ext = dealii::GeometryInfo<dim>::unit_normal_vector[face_subface_ext.first];

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    //const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    //const real2 cell_diameter_int = fe_values_int.get_cell()->diameter();
    //const real2 cell_diameter_ext = fe_values_ext.get_cell()->diameter();
    //const real2 artificial_diss_coeff_int = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
    //                                        this->discontinuity_sensor(cell_diameter_int, soln_coeff_int, fe_values_int.get_fe())
    //                                        : 0.0;
    //const real2 artificial_diss_coeff_ext = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
    //                                        this->discontinuity_sensor(cell_diameter_ext, soln_coeff_ext, fe_values_ext.get_fe())
    //                                        : 0.0;
    const real2 artificial_diss_coeff_int = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
                                            this->artificial_dissipation_coeffs[current_cell_index]
                                            : 0.0;
    const real2 artificial_diss_coeff_ext = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
                                            this->artificial_dissipation_coeffs[neighbor_cell_index]
                                            : 0.0;

    (void) artificial_diss_coeff_int;
    (void) artificial_diss_coeff_ext;
    typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
        this->triangulation.get(), cell->level(), cell->index(), &(this->dof_handler_artificial_dissipation));
    const unsigned int n_dofs_arti_diss = this->fe_q_artificial_dissipation.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);
    artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);

    std::vector<real> artificial_diss_coeff_at_q(n_face_quad_pts);
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        artificial_diss_coeff_at_q[iquad] = 0.0;

        if ( this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ) {
            const dealii::Point<dim,real> point = unit_quad_pts_int[iquad];
            for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
                const unsigned int index = dof_indices_artificial_dissipation[idof];
                artificial_diss_coeff_at_q[iquad] += this->artificial_dissipation_c0[index] * this->fe_q_artificial_dissipation.shape_value(idof, point);
            }
        }
        artificial_diss_coeff_at_q[iquad] = 0.0;
    }

    std::vector<real2> jacobian_determinant_int(n_face_quad_pts);
    std::vector<real2> jacobian_determinant_ext(n_face_quad_pts);
    std::vector<Tensor2D> jacobian_transpose_inverse_int(n_face_quad_pts);
    std::vector<Tensor2D> jacobian_transpose_inverse_ext(n_face_quad_pts);

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        if (compute_metric_derivatives) {
            jacobian_determinant_int[iquad] = dealii::determinant(metric_jac_int[iquad]);
            jacobian_determinant_ext[iquad] = dealii::determinant(metric_jac_ext[iquad]);

            jacobian_transpose_inverse_int[iquad] = dealii::transpose(dealii::invert(metric_jac_int[iquad]));
            jacobian_transpose_inverse_ext[iquad] = dealii::transpose(dealii::invert(metric_jac_ext[iquad]));
        }
    }

#ifdef KOPRIVA_METRICS_FACE
    auto old_jacobian_determinant_int = jacobian_determinant_int;
    auto old_jacobian_determinant_ext = jacobian_determinant_ext;
    auto old_jacobian_transpose_inverse_int = jacobian_transpose_inverse_int;
    auto old_jacobian_transpose_inverse_ext = jacobian_transpose_inverse_ext;

    if constexpr (dim != 1) {
        evaluate_covariant_metric_jacobian<dim,real2> ( face_quadrature_int, coords_coeff_int, fe_metric, jacobian_transpose_inverse_int, jacobian_determinant_int);
        evaluate_covariant_metric_jacobian<dim,real2> ( face_quadrature_ext, coords_coeff_ext, fe_metric, jacobian_transpose_inverse_ext, jacobian_determinant_ext);
    }

    //for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
    //    //if (abs(old_jacobian_determinant_int[iquad] - jacobian_determinant_int[iquad]) > 1e-10) {
    //        std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    //        std::cout << "Not the same jac det int, iquad " << iquad << std::endl;
    //        std::cout << old_jacobian_determinant_int[iquad] << std::endl;
    //        std::cout << jacobian_determinant_int[iquad] << std::endl;
    //    //}
    //    //if (abs(old_jacobian_determinant_ext[iquad] - jacobian_determinant_ext[iquad]) > 1e-10) {
    //        std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    //        std::cout << "Not the same jac det ext, iquad " << iquad << std::endl;
    //        std::cout << old_jacobian_determinant_ext[iquad] << std::endl;
    //        std::cout << jacobian_determinant_ext[iquad] << std::endl;
    //    //}
    //    for (int row=0;row<dim;++row) {
    //        for (int col=0;col<dim;++col) {
    //            //if (abs(old_jacobian_transpose_inverse_int[iquad][row][col] - jacobian_transpose_inverse_int[iquad][row][col]) > 1e-10) {
    //                std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    //                std::cout << "Not the same jac inv tran int, iquad " << iquad << " row " << row << " col " << col << std::endl;
    //                std::cout << old_jacobian_transpose_inverse_int[iquad][row][col] << std::endl;
    //                std::cout << jacobian_transpose_inverse_int[iquad][row][col] << std::endl;
    //            }
    //        //}
    //    }
    //    for (int row=0;row<dim;++row) {
    //        for (int col=0;col<dim;++col) {
    //            //if (abs(old_jacobian_transpose_inverse_ext[iquad][row][col] - jacobian_transpose_inverse_ext[iquad][row][col]) > 1e-10) {
    //                std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    //                std::cout << "Not the same jac inv tran ext, iquad " << iquad << " row " << row << " col " << col << std::endl;
    //                std::cout << old_jacobian_transpose_inverse_ext[iquad][row][col] << std::endl;
    //                std::cout << jacobian_transpose_inverse_ext[iquad][row][col] << std::endl;
    //            //}
    //        }
    //    }
    //}
#endif

    // Note: This is ignored when use_periodic_bc is set to true -- this variable has no other function when dim!=1
    if(this->all_parameters->use_periodic_bc == false) {
        check_same_coords<dim,real2>(unit_quad_pts_int, unit_quad_pts_ext, coords_coeff_int, coords_coeff_ext, fe_metric, 1e-10);
    }

    // Compute metrics
    std::vector<Tensor1D> phys_unit_normal_int(n_face_quad_pts), phys_unit_normal_ext(n_face_quad_pts);
    std::vector<real2> surface_jac_det(n_face_quad_pts);
    std::vector<real2> faceJxW(n_face_quad_pts);

    dealii::FullMatrix<real> interpolation_operator_int(n_soln_dofs_int, n_face_quad_pts);
    dealii::FullMatrix<real> interpolation_operator_ext(n_soln_dofs_ext, n_face_quad_pts);
    std::array<dealii::FullMatrix<real2>,dim> gradient_operator_int, gradient_operator_ext;
    for (int d=0;d<dim;++d) {
        gradient_operator_int[d].reinit(dealii::TableIndices<2>(n_soln_dofs_int, n_face_quad_pts));
        gradient_operator_ext[d].reinit(dealii::TableIndices<2>(n_soln_dofs_ext, n_face_quad_pts));
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        real2 surface_jac_det_int, surface_jac_det_ext;

        if (compute_metric_derivatives) {

            const real2 jac_det_int = jacobian_determinant_int[iquad];
            const real2 jac_det_ext = jacobian_determinant_ext[iquad];

            const Tensor2D jac_inv_tran_int = jacobian_transpose_inverse_int[iquad];
            const Tensor2D jac_inv_tran_ext = jacobian_transpose_inverse_ext[iquad];

            const Tensor1D normal_int = vmult(jac_inv_tran_int, unit_normal_int);
            const Tensor1D normal_ext = vmult(jac_inv_tran_ext, unit_normal_ext);
            const real2 area_int = norm(normal_int);
            const real2 area_ext = norm(normal_ext);

            // Technically the normals have jac_det multiplied.
            // However, we use normalized normals by convention, so the the term
            // ends up appearing in the surface jacobian.

            for (int d=0;d<dim;++d) {
                phys_unit_normal_int[iquad][d] = normal_int[d] / area_int;
            }
            for (int d=0;d<dim;++d) {
                phys_unit_normal_ext[iquad][d] = normal_ext[d] / area_ext;
            }

            surface_jac_det_int = area_int*jac_det_int;
            surface_jac_det_ext = area_ext*jac_det_ext;


            if (std::is_same<double,real2>::value) {
                bool valid_metrics = true;
                // surface_jac_det is the 'volume' compression/expansion of the face w.r.t. the reference cell,
                // analogous to volume jacobian determinant.
                //
                // When the cells have the same coarseness, their surface Jacobians must be the same.
                //
                // When the cells do not have the same coarseness, their surface Jacobians will not be the same.
                // Therefore, we must use the Jacobians coming from the smaller face since it accurately represents
                // the surface area being integrated.
                if (face_subface_int.second == -1 && face_subface_ext.second == -1) {
                    if(abs(surface_jac_det_int-surface_jac_det_ext) > 1e-12) {
                        std::cout << std::endl;
                        std::cout << "iquad " << iquad << " Non-matching surface jacobians "
                            << surface_jac_det_int << " " << surface_jac_det_ext<< std::endl;

                        assert(abs(surface_jac_det_int-surface_jac_det_ext) < 1e-12);
                        valid_metrics = false;
                    }
                }
                real2 diff_norm = 0;
                for (int d=0;d<dim;++d) {
                    const real2 diff = phys_unit_normal_int[iquad][d]+phys_unit_normal_ext[iquad][d];
                    diff_norm += diff*diff;
                }
                diff_norm = sqrt(diff_norm);
                if (diff_norm > 1e-10) {
                    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
                    std::cout << "Non-matching normals. Error norm: " << diff_norm << std::endl;
                    for (int d=0;d<dim;++d) {
                        //assert(abs(phys_unit_normal_int[iquad][d]+phys_unit_normal_ext[iquad][d]) < 1e-10);
                        std::cout << " normal_int["<<d<<"] : " << phys_unit_normal_int[iquad][d] 
                                  << " normal_ext["<<d<<"] : " << phys_unit_normal_ext[iquad][d]
                                  << std::endl;
                    }
                    valid_metrics = false;
                }
                if (!valid_metrics) {
                    //for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
                    //   rhs_int[itest_int] += 1e20;
                    //}
                    //for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
                    //   rhs_ext[itest_ext] += 1e20;
                    //}
                }

            }
            //phys_unit_normal_ext[iquad] = -phys_unit_normal_int[iquad];//normal_ext / area_ext; Must use opposite normal to be consistent with explicit

            for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
                interpolation_operator_int[idof][iquad] = fe_int.shape_value(idof,unit_quad_pts_int[iquad]);
                dealii::Tensor<1,dim,real> ref_shape_grad = fe_int.shape_grad(idof,unit_quad_pts_int[iquad]);
                const Tensor1D phys_shape_grad = vmult(jac_inv_tran_int, ref_shape_grad);
                for (int d=0;d<dim;++d) {
                    gradient_operator_int[d][idof][iquad] = phys_shape_grad[d];
                }
            }
            for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
                interpolation_operator_ext[idof][iquad] = fe_ext.shape_value(idof,unit_quad_pts_ext[iquad]);
                dealii::Tensor<1,dim,real> ref_shape_grad = fe_ext.shape_grad(idof,unit_quad_pts_ext[iquad]);
                const Tensor1D phys_shape_grad = vmult(jac_inv_tran_ext, ref_shape_grad);
                for (int d=0;d<dim;++d) {
                    gradient_operator_ext[d][idof][iquad] = phys_shape_grad[d];
                }
            }

        } else {
            for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
                interpolation_operator_int[idof][iquad] = fe_int.shape_value(idof,unit_quad_pts_int[iquad]);
            }
            for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
                interpolation_operator_ext[idof][iquad] = fe_ext.shape_value(idof,unit_quad_pts_ext[iquad]);
            }
            for (int d=0;d<dim;++d) {
                for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
                    const unsigned int istate = fe_int.system_to_component_index(idof).first;
                    gradient_operator_int[d][idof][iquad] = fe_values_int.shape_grad_component(idof, iquad, istate)[d];
                }
                for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
                    const unsigned int istate = fe_ext.system_to_component_index(idof).first;
                    gradient_operator_ext[d][idof][iquad] = fe_values_ext.shape_grad_component(idof, iquad, istate)[d];
                }
            }
            surface_jac_det_int = fe_values_int.JxW(iquad)/face_quadrature_int.weight(iquad);
            surface_jac_det_ext = fe_values_ext.JxW(iquad)/face_quadrature_ext.weight(iquad);

            phys_unit_normal_int[iquad] = fe_values_int.normal_vector(iquad);
            phys_unit_normal_ext[iquad] = -phys_unit_normal_int[iquad]; // Must use opposite normal to be consistent with explicit
        }
        // When the cells do not have the same coarseness, their surface Jacobians will not be the same.
        // Therefore, we must use the Jacobians coming from the smaller face since it accurately represents
        // the surface area being computed.
        //
        // Note that it is possible for the smaller cell to have larger surface Jacobians than the larger cell,
        // but not at the same physical location.
        if ( surface_jac_det_int > surface_jac_det_ext) {
            // Interior is the large face.
            // Exterior is the small face.
            surface_jac_det[iquad] = surface_jac_det_ext;
            //phys_unit_normal_ext[iquad] = -phys_unit_normal_int[iquad];
        } else {
            // Exterior is the large face.
            // Interior is the small face.
            surface_jac_det[iquad] = surface_jac_det_int;
            //phys_unit_normal_int[iquad] = -phys_unit_normal_ext[iquad];
        }

        faceJxW[iquad] = surface_jac_det[iquad] * face_quadrature_int.weight(iquad);
    }


    // Interpolate solution
    std::vector<ADArray> soln_int(n_face_quad_pts), soln_ext(n_face_quad_pts);
    evaluate_finite_element_values<dim, real2, nstate> (unit_quad_pts_int, soln_coeff_int, fe_int, soln_int);
    evaluate_finite_element_values<dim, real2, nstate> (unit_quad_pts_ext, soln_coeff_ext, fe_ext, soln_ext);

    // Interpolate solution gradient
    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts), soln_grad_ext(n_face_quad_pts);
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        for (int istate=0; istate<nstate; istate++) {
            soln_grad_int[iquad][istate] = 0;
            soln_grad_ext[iquad][istate] = 0;
        }

        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
            const unsigned int istate = fe_int.system_to_component_index(idof).first;
            for (int d=0;d<dim;++d) {
                soln_grad_int[iquad][istate][d] += soln_coeff_int[idof] * gradient_operator_int[d][idof][iquad];
            }
        }
        for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
            const unsigned int istate = fe_ext.system_to_component_index(idof).first;
            for (int d=0;d<dim;++d) {
                soln_grad_ext[iquad][istate][d] += soln_coeff_ext[idof] * gradient_operator_ext[d][idof][iquad];
            }
        }
    }

    // Assemble BR2 gradient correction right-hand side

    using DissFlux = Parameters::AllParameters::DissipativeNumericalFlux;
    if (this->all_parameters->diss_num_flux_type == DissFlux::bassi_rebay_2) {

        const dealii::FiniteElement<dim> &base_fe_int = fe_int.get_sub_fe(0,1);
        const dealii::FiniteElement<dim> &base_fe_ext = fe_ext.get_sub_fe(0,1);
        const unsigned int n_base_dofs_int = base_fe_int.n_dofs_per_cell();
        const unsigned int n_base_dofs_ext = base_fe_ext.n_dofs_per_cell();

        // Obtain solution jump
        std::vector<std::array<dealii::Tensor<1,dim,real2>, nstate>> soln_jump_int(n_face_quad_pts);
        std::vector<std::array<dealii::Tensor<1,dim,real2>, nstate>> soln_jump_ext(n_face_quad_pts);
        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            for (int s=0; s<nstate; s++) {
                for (int d=0; d<dim; d++) {
                    soln_jump_int[iquad][s][d] = (soln_int[iquad][s] - soln_ext[iquad][s]) * phys_unit_normal_int[iquad][d];
                    soln_jump_ext[iquad][s][d] = (soln_ext[iquad][s] - soln_int[iquad][s]) * (-phys_unit_normal_int[iquad][d]);
                }
            }
        }


        // RHS of R lifting operator.
        std::vector<ADArray> lifting_op_R_rhs_int(n_base_dofs_int);
        for (unsigned int idof_base=0; idof_base<n_base_dofs_int; ++idof_base) {
            for (int s=0; s<nstate; s++) {

                const unsigned int idof = fe_int.component_to_system_index(s, idof_base);
                lifting_op_R_rhs_int[idof_base][s] = 0.0;

                for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

                    for (int d=0; d<dim; ++d) {
                        //const real2 basis_average = 0.5 * (gradient_operator_int[d][idof][iquad] + 0.0);
                        const double basis_average = 0.5 * (interpolation_operator_int[idof][iquad] + 0.0);
                        lifting_op_R_rhs_int[idof_base][s] -= soln_jump_int[iquad][s][d] * basis_average * faceJxW[iquad];
                    }
                }

            }
        }

        std::vector<ADArray> lifting_op_R_rhs_ext(n_base_dofs_ext);
        for (unsigned int idof_base=0; idof_base<n_base_dofs_ext; ++idof_base) {
            for (int s=0; s<nstate; s++) {

                const unsigned int idof = fe_ext.component_to_system_index(s, idof_base);
                lifting_op_R_rhs_ext[idof_base][s] = 0.0;

                for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

                    for (int d=0; d<dim; ++d) {
                        //const real2 basis_average = 0.5 * ( 0.0 + gradient_operator_ext[d][idof][iquad] );
                        const double basis_average = 0.5 * ( 0.0 + interpolation_operator_ext[idof][iquad] );
                        lifting_op_R_rhs_ext[idof_base][s] -= soln_jump_ext[iquad][s][d] * basis_average * faceJxW[iquad];
                    }
                }

            }
        }

        std::vector<ADArray> soln_grad_corr_int(n_base_dofs_int), soln_grad_corr_ext(n_base_dofs_ext);
        compute_br2_correction<dim,nstate,real2>(fe_int, coords_coeff_int, fe_metric, lifting_op_R_rhs_int, soln_grad_corr_int);
        compute_br2_correction<dim,nstate,real2>(fe_ext, coords_coeff_ext, fe_metric, lifting_op_R_rhs_ext, soln_grad_corr_ext);

        correct_the_gradient<dim,nstate,real2>( soln_grad_corr_int, fe_int, soln_jump_int, interpolation_operator_int, gradient_operator_int, soln_grad_int);
        correct_the_gradient<dim,nstate,real2>( soln_grad_corr_ext, fe_ext, soln_jump_ext, interpolation_operator_ext, gradient_operator_ext, soln_grad_ext);

    }


    ADArray conv_num_flux_dot_n;
    ADArray diss_soln_num_flux; // u*
    ADArray diss_auxi_num_flux_dot_n; // sigma*

    ADArrayTensor1 diss_flux_jump_int; // u*-u_int
    ADArrayTensor1 diss_flux_jump_ext; // u*-u_ext

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        // Evaluate physical convective flux, physical dissipative flux, and source term
        conv_num_flux_dot_n = conv_num_flux.evaluate_flux(soln_int[iquad], soln_ext[iquad], phys_unit_normal_int[iquad]);
        diss_soln_num_flux = diss_num_flux.evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], phys_unit_normal_int[iquad]);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            for (int d=0; d<dim; d++) {
                diss_soln_jump_int[s][d] = (diss_soln_num_flux[s] - soln_int[iquad][s]) * phys_unit_normal_int[iquad][d];
                diss_soln_jump_ext[s][d] = (diss_soln_num_flux[s] - soln_ext[iquad][s]) * phys_unit_normal_ext[iquad][d];
            }
        }
        diss_flux_jump_int = physics.dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext = physics.dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

        if (this->all_parameters->artificial_dissipation_param.add_artificial_dissipation) {
            //const ADArrayTensor1 artificial_diss_flux_jump_int = physics.artificial_dissipative_flux (artificial_diss_coeff_int, soln_int[iquad], diss_soln_jump_int);
            //const ADArrayTensor1 artificial_diss_flux_jump_ext = physics.artificial_dissipative_flux (artificial_diss_coeff_ext, soln_ext[iquad], diss_soln_jump_ext);
            const ADArrayTensor1 artificial_diss_flux_jump_int =  DGBaseState<dim,nstate,real,MeshType>::artificial_dissip->calc_artificial_dissipation_flux(soln_int[iquad], diss_soln_jump_int,artificial_diss_coeff_at_q[iquad]);
            const ADArrayTensor1 artificial_diss_flux_jump_ext =  DGBaseState<dim,nstate,real,MeshType>::artificial_dissip->calc_artificial_dissipation_flux(soln_ext[iquad], diss_soln_jump_ext,artificial_diss_coeff_at_q[iquad]);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[s] += artificial_diss_flux_jump_int[s];
                diss_flux_jump_ext[s] += artificial_diss_flux_jump_ext[s];
            }
        }


        diss_auxi_num_flux_dot_n = diss_num_flux.evaluate_auxiliary_flux(
            //artificial_diss_coeff_int,
            //artificial_diss_coeff_ext,
            artificial_diss_coeff_at_q[iquad],
            artificial_diss_coeff_at_q[iquad],
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            phys_unit_normal_int[iquad], penalty);

        // From test functions associated with interior cell point of view
        for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
            real2 rhs = 0.0;
            const unsigned int istate = fe_int.system_to_component_index(itest_int).first;

            const real2 JxW_iquad = faceJxW[iquad];
            // Convection
            rhs = rhs - interpolation_operator_int[itest_int][iquad] * conv_num_flux_dot_n[istate] * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_int[itest_int][iquad] * diss_auxi_num_flux_dot_n[istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_int[d][itest_int][iquad] * diss_flux_jump_int[istate][d] * JxW_iquad;
            }

            rhs_int[itest_int] += rhs;
            dual_dot_residual += dual_int[itest_int]*rhs;
        }

        // From test functions associated with neighbor cell point of view
        for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
            real2 rhs = 0.0;
            const unsigned int istate = fe_ext.system_to_component_index(itest_ext).first;

            const real2 JxW_iquad = faceJxW[iquad];
            // Convection
            rhs = rhs - interpolation_operator_ext[itest_ext][iquad] * (-conv_num_flux_dot_n[istate]) * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_ext[itest_ext][iquad] * (-diss_auxi_num_flux_dot_n[istate]) * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_ext[d][itest_ext][iquad] * diss_flux_jump_ext[istate][d] * JxW_iquad;
            }

            rhs_ext[itest_ext] += rhs;
            dual_dot_residual += dual_ext[itest_ext]*rhs;
        }
    } // Quadrature point loop

}

#ifdef FADFAD
template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_face_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const std::pair<unsigned int, int> face_subface_int,
    const std::pair<unsigned int, int> face_subface_ext,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_int,
    const dealii::FESystem<dim,dim> &fe_ext,
    const dealii::Quadrature<dim-1> &face_quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;
    using adtype = FadFadType;
    using ADArray = std::array<adtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,adtype>, nstate >;

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_soln_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_ext.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_soln_dofs_ext, soln_dof_indices_ext.size());

    std::vector< adtype > coords_coeff_int(n_metric_dofs);
    std::vector< adtype > coords_coeff_ext(n_metric_dofs);
    std::vector< adtype > soln_coeff_int(n_soln_dofs_int);
    std::vector< adtype > soln_coeff_ext(n_soln_dofs_ext);

    // Current derivative ordering is: soln_int, soln_ext, metric_int, metric_ext
    unsigned int w_int_start, w_int_end, w_ext_start, w_ext_end,
                 x_int_start, x_int_end, x_ext_start, x_ext_end;
    automatic_differentiation_indexing_2(
        compute_dRdW, compute_dRdX, compute_d2R,
        n_soln_dofs_int, n_soln_dofs_ext, n_metric_dofs,
        w_int_start, w_int_end, w_ext_start, w_ext_end,
        x_int_start, x_int_end, x_ext_start, x_ext_end);

    const unsigned int n_total_indep = x_ext_end;
    unsigned int i_derivative = 0;

    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        const real val = this->solution(soln_dof_indices_int[idof]);
        soln_coeff_int[idof] = val;
        soln_coeff_int[idof].val() = val;

        if (compute_dRdW || compute_d2R) soln_coeff_int[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) soln_coeff_int[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdW || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
        const real val = this->solution(soln_dof_indices_ext[idof]);
        soln_coeff_ext[idof] = val;
        soln_coeff_ext[idof].val() = val;

        if (compute_dRdW || compute_d2R) soln_coeff_ext[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) soln_coeff_ext[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdW || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices_int[idof]];
        coords_coeff_int[idof] = val;
        coords_coeff_int[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff_int[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff_int[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdX || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices_ext[idof]];
        coords_coeff_ext[idof] = val;
        coords_coeff_ext[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff_ext[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff_ext[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdX || compute_d2R) i_derivative++;
    }
    AssertDimension(i_derivative, n_total_indep);

    std::vector<double> dual_int(n_soln_dofs_int);
    std::vector<double> dual_ext(n_soln_dofs_ext);

    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_int[itest];
        dual_int[itest] = this->dual[global_residual_row];
    }
    for (unsigned int itest=0; itest<n_soln_dofs_ext; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_ext[itest];
        dual_ext[itest] = this->dual[global_residual_row];
    }

    std::vector<adtype> rhs_int(n_soln_dofs_int);
    std::vector<adtype> rhs_ext(n_soln_dofs_ext);
    adtype dual_dot_residual;

    const auto &physics = *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_fad_fad);
    const auto &conv_num_flux = *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad_fad);
    const auto &diss_num_flux = *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad_fad);
    assemble_face_term(
        cell,
        current_cell_index,
        neighbor_cell_index,
        soln_coeff_int,
        soln_coeff_ext,
        coords_coeff_int,
        coords_coeff_ext,
        dual_int,
        dual_ext,
        face_subface_int,
        face_subface_ext,
        face_data_set_int,
        face_data_set_ext,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_int,
        fe_values_ext,
        penalty,
        fe_int,
        fe_ext,
        fe_metric,
        face_quadrature_int,
        face_quadrature_ext,
        rhs_int,
        rhs_ext,
        dual_dot_residual,
        compute_dRdW, compute_dRdX, compute_d2R);

    for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
        local_rhs_int_cell[itest_int] += rhs_int[itest_int].val().val();
    }
    for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
        local_rhs_ext_cell[itest_ext] += rhs_ext[itest_ext].val().val();
    }

    if (compute_dRdW) {
        std::vector<real> residual_derivatives(n_soln_dofs_int);
        for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
            // dR_int_dW_int
            residual_derivatives.resize(n_soln_dofs_int);
            for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                const unsigned int i_dx = idof+w_int_start;
                residual_derivatives[idof] = rhs_int[itest_int].dx(i_dx).val();
            }
            const bool elide_zero_values = false;
            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_int, residual_derivatives, elide_zero_values);

            // dR_int_dW_ext
            residual_derivatives.resize(n_soln_dofs_ext);
            for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                const unsigned int i_dx = idof+w_ext_start;
                residual_derivatives[idof] = rhs_int[itest_int].dx(i_dx).val();
            }
            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_ext, residual_derivatives, elide_zero_values);
        }

        for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
            // dR_ext_dW_int
            residual_derivatives.resize(n_soln_dofs_int);
            for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                const unsigned int i_dx = idof+w_int_start;
                residual_derivatives[idof] = rhs_ext[itest_ext].dx(i_dx).val();
            }
            const bool elide_zero_values = false;
            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_int, residual_derivatives, elide_zero_values);

            // dR_ext_dW_ext
            residual_derivatives.resize(n_soln_dofs_ext);
            for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                const unsigned int i_dx = idof+w_ext_start;
                residual_derivatives[idof] = rhs_ext[itest_ext].dx(i_dx).val();
            }
            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_ext, residual_derivatives, elide_zero_values);
        }
    }
    if (compute_dRdX) {
        std::vector<real> residual_derivatives(n_metric_dofs);
        for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
            // dR_int_dX_int
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_int_start;
                residual_derivatives[idof] = rhs_int[itest_int].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_int, residual_derivatives);

            // dR_int_dX_ext
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_ext_start;
                residual_derivatives[idof] = rhs_int[itest_int].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_ext, residual_derivatives);
        }
        for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
            // dR_ext_dX_int
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_int_start;
                residual_derivatives[idof] = rhs_ext[itest_ext].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_int, residual_derivatives);

            // dR_ext_dX_ext
            // residual_derivatives.resize(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_ext_start;
                residual_derivatives[idof] = rhs_ext[itest_ext].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_ext, residual_derivatives);
        }
    }

    if (compute_d2R) {
        std::vector<real> dWidWint(n_soln_dofs_int);
        std::vector<real> dWidWext(n_soln_dofs_ext);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);
        // dWint
        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {

            const unsigned int i_dx = idof+w_int_start;
            const FadType dWi = dual_dot_residual.dx(i_dx);

            // dWint_dWint
            for (unsigned int jdof=0; jdof<n_soln_dofs_int; ++jdof) {
                const unsigned int j_dx = jdof+w_int_start;
                dWidWint[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_int[idof], soln_dof_indices_int, dWidWint);

            // dWint_dWext
            for (unsigned int jdof=0; jdof<n_soln_dofs_ext; ++jdof) {
                const unsigned int j_dx = jdof+w_ext_start;
                dWidWext[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_int[idof], soln_dof_indices_ext, dWidWext);

            // dWint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_int[idof], metric_dof_indices_int, dWidX);

            // dWint_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_int[idof], metric_dof_indices_ext, dWidX);
        }
        // dWext
        for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {

            const unsigned int i_dx = idof+w_ext_start;
            const FadType dWi = dual_dot_residual.dx(i_dx);

            // dWext_dWint
            for (unsigned int jdof=0; jdof<n_soln_dofs_int; ++jdof) {
                const unsigned int j_dx = jdof+w_int_start;
                dWidWint[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_ext[idof], soln_dof_indices_int, dWidWint);

            // dWext_dWext
            for (unsigned int jdof=0; jdof<n_soln_dofs_ext; ++jdof) {
                const unsigned int j_dx = jdof+w_ext_start;
                dWidWext[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_ext[idof], soln_dof_indices_ext, dWidWext);

            // dWext_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_ext[idof], metric_dof_indices_int, dWidX);

            // dWext_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_ext[idof], metric_dof_indices_ext, dWidX);
        }

        // dXint
        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_int_start;
            const FadType dWi = dual_dot_residual.dx(i_dx);

            // dXint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_int[idof], metric_dof_indices_int, dWidX);

            // dXint_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_int[idof], metric_dof_indices_ext, dWidX);
        }
        // dXext
        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_ext_start;
            const FadType dWi = dual_dot_residual.dx(i_dx);

            // dXext_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_ext[idof], metric_dof_indices_int, dWidX);

            // dXext_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_ext[idof], metric_dof_indices_ext, dWidX);
        }
    }
}
#endif

#ifndef FADFAD
template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGWeak<dim,nstate,real,MeshType>::assemble_face_codi_taped_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const std::pair<unsigned int, int> face_subface_int,
    const std::pair<unsigned int, int> face_subface_ext,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_int,
    const dealii::FESystem<dim,dim> &fe_ext,
    const dealii::Quadrature<dim-1> &face_quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    const Physics::PhysicsBase<dim, nstate, adtype> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    using ADArray = std::array<adtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,adtype>, nstate >;

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_soln_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_ext.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_soln_dofs_ext, soln_dof_indices_ext.size());

    std::vector< adtype > coords_coeff_int(n_metric_dofs);
    std::vector< adtype > coords_coeff_ext(n_metric_dofs);
    std::vector< adtype > soln_coeff_int(n_soln_dofs_int);
    std::vector< adtype > soln_coeff_ext(n_soln_dofs_ext);

    // Current derivative ordering is: soln_int, soln_ext, metric_int, metric_ext
    unsigned int w_int_start, w_int_end, w_ext_start, w_ext_end,
                 x_int_start, x_int_end, x_ext_start, x_ext_end;
    automatic_differentiation_indexing_2(
        compute_dRdW, compute_dRdX, compute_d2R,
        n_soln_dofs_int, n_soln_dofs_ext, n_metric_dofs,
        w_int_start, w_int_end, w_ext_start, w_ext_end,
        x_int_start, x_int_end, x_ext_start, x_ext_end);

    using TH = codi::TapeHelper<adtype>;
    TH th;
    adtype::getGlobalTape();
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.startRecording();
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        const real val = this->solution(soln_dof_indices_int[idof]);
        soln_coeff_int[idof] = val;
        if (compute_dRdW || compute_d2R) {
            th.registerInput(soln_coeff_int[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(soln_coeff_int[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
        const real val = this->solution(soln_dof_indices_ext[idof]);
        soln_coeff_ext[idof] = val;
        if (compute_dRdW || compute_d2R) {
            th.registerInput(soln_coeff_ext[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(soln_coeff_ext[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices_int[idof]];
        coords_coeff_int[idof] = val;
        if (compute_dRdX || compute_d2R) {
            th.registerInput(coords_coeff_int[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(coords_coeff_int[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices_ext[idof]];
        coords_coeff_ext[idof] = val;
        if (compute_dRdX || compute_d2R) {
            th.registerInput(coords_coeff_ext[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(coords_coeff_ext[idof]);
        }
    }

    std::vector<double> dual_int(n_soln_dofs_int);
    std::vector<double> dual_ext(n_soln_dofs_ext);

    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_int[itest];
        dual_int[itest] = this->dual[global_residual_row];
    }
    for (unsigned int itest=0; itest<n_soln_dofs_ext; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_ext[itest];
        dual_ext[itest] = this->dual[global_residual_row];
    }

    std::vector<adtype> rhs_int(n_soln_dofs_int);
    std::vector<adtype> rhs_ext(n_soln_dofs_ext);
    adtype dual_dot_residual;

    assemble_face_term(
        cell,
        current_cell_index,
        neighbor_cell_index,
        soln_coeff_int,
        soln_coeff_ext,
        coords_coeff_int,
        coords_coeff_ext,
        dual_int,
        dual_ext,
        face_subface_int,
        face_subface_ext,
        face_data_set_int,
        face_data_set_ext,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_int,
        fe_values_ext,
        penalty,
        fe_int,
        fe_ext,
        fe_metric,
        face_quadrature,
        rhs_int,
        rhs_ext,
        dual_dot_residual,
        compute_dRdW, compute_dRdX, compute_d2R);

    if (compute_dRdW || compute_dRdX) {
        for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {
            th.registerOutput(rhs_int[itest]);
        }
        for (unsigned int itest=0; itest<n_soln_dofs_ext; ++itest) {
            th.registerOutput(rhs_ext[itest]);
        }
    } else if (compute_d2R) {
        th.registerOutput(dual_dot_residual);
    }
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.stopRecording();
        //adtype::getGlobalTape().printStatistics();
    }

    for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
        local_rhs_int_cell[itest_int] += getValue<adtype>(rhs_int[itest_int]);
    }
    for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
        local_rhs_ext_cell[itest_ext] += getValue<adtype>(rhs_ext[itest_ext]);
    }

    if (compute_dRdW || compute_dRdX) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);

        if (compute_dRdW) {
            std::vector<real> residual_derivatives(n_soln_dofs_int);

            for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
                int i_dependent = itest_int;

                // dR_int_dW_int
                residual_derivatives.resize(n_soln_dofs_int);
                for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                    const unsigned int i_dx = idof+w_int_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                const bool elide_zero_values = false;
                this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_int, residual_derivatives, elide_zero_values);

                // dR_int_dW_ext
                residual_derivatives.resize(n_soln_dofs_ext);
                for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                    const unsigned int i_dx = idof+w_ext_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_ext, residual_derivatives, elide_zero_values);
            }

            for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {

                int i_dependent = n_soln_dofs_int + itest_ext;

                // dR_ext_dW_int
                residual_derivatives.resize(n_soln_dofs_int);
                for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                    const unsigned int i_dx = idof+w_int_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                const bool elide_zero_values = false;
                this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_int, residual_derivatives, elide_zero_values);

                // dR_ext_dW_ext
                residual_derivatives.resize(n_soln_dofs_ext);
                for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                    const unsigned int i_dx = idof+w_ext_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_ext, residual_derivatives, elide_zero_values);
            }
        }

        if (compute_dRdX) {
            std::vector<real> residual_derivatives(n_metric_dofs);

            for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {

                int i_dependent = itest_int;

                // dR_int_dX_int
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_int_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_int, residual_derivatives);

                // dR_int_dX_ext
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_ext_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_ext, residual_derivatives);
            }

            for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {

                int i_dependent = n_soln_dofs_int + itest_ext;

                // dR_ext_dX_int
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_int_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_int, residual_derivatives);

                // dR_ext_dX_ext
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_ext_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_ext, residual_derivatives);
            }
        }

        th.deleteJacobian(jac);
    }

    if (compute_d2R) {
        typename TH::HessianType& hes = th.createHessian();
        th.evalHessian(hes);

        std::vector<real> dWidW(n_soln_dofs_int);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);

        int i_dependent = (compute_dRdW || compute_dRdX) ? n_soln_dofs_int + n_soln_dofs_ext : 0;

        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {

            const unsigned int i_dx = idof+w_int_start;

            // dWint_dWint
            for (unsigned int jdof=0; jdof<n_soln_dofs_int; ++jdof) {
                const unsigned int j_dx = jdof+w_int_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_int[idof], soln_dof_indices_int, dWidW);

            // dWint_dWext
            for (unsigned int jdof=0; jdof<n_soln_dofs_ext; ++jdof) {
                const unsigned int j_dx = jdof+w_ext_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_int[idof], soln_dof_indices_ext, dWidW);

            // dWint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_int[idof], metric_dof_indices_int, dWidX);

            // dWint_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_int[idof], metric_dof_indices_ext, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_int_start;

            // dXint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_int[idof], metric_dof_indices_int, dXidX);

            // dXint_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_int[idof], metric_dof_indices_ext, dXidX);
        }

        dWidW.resize(n_soln_dofs_ext);

        for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {

            const unsigned int i_dx = idof+w_ext_start;

            // dWext_dWint
            for (unsigned int jdof=0; jdof<n_soln_dofs_int; ++jdof) {
                const unsigned int j_dx = jdof+w_int_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_ext[idof], soln_dof_indices_int, dWidW);

            // dWext_dWext
            for (unsigned int jdof=0; jdof<n_soln_dofs_ext; ++jdof) {
                const unsigned int j_dx = jdof+w_ext_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_ext[idof], soln_dof_indices_ext, dWidW);

            // dWext_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_ext[idof], metric_dof_indices_int, dWidX);

            // dWext_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_ext[idof], metric_dof_indices_ext, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_ext_start;

            // dXint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_ext[idof], metric_dof_indices_int, dXidX);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_ext[idof], metric_dof_indices_ext, dXidX);
        }

        th.deleteHessian(hes);
    }

    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        adtype::getGlobalTape().deactivateValue(soln_coeff_int[idof]);
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
        adtype::getGlobalTape().deactivateValue(soln_coeff_ext[idof]);
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        adtype::getGlobalTape().deactivateValue(coords_coeff_int[idof]);
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        adtype::getGlobalTape().deactivateValue(coords_coeff_ext[idof]);
    }

}

template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_face_residual(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const std::pair<unsigned int, int> face_subface_int,
    const std::pair<unsigned int, int> face_subface_ext,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_int,
    const dealii::FESystem<dim,dim> &fe_ext,
    const dealii::Quadrature<dim-1> &face_quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    const Physics::PhysicsBase<dim, nstate, real> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, real> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, real> &diss_num_flux,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    using ADArray = std::array<real,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_soln_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_ext.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_soln_dofs_ext, soln_dof_indices_ext.size());

    std::vector< real > coords_coeff_int(n_metric_dofs);
    std::vector< real > coords_coeff_ext(n_metric_dofs);
    std::vector< real > soln_coeff_int(n_soln_dofs_int);
    std::vector< real > soln_coeff_ext(n_soln_dofs_ext);

    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        const real val = this->solution(soln_dof_indices_int[idof]);
        soln_coeff_int[idof] = val;
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
        const real val = this->solution(soln_dof_indices_ext[idof]);
        soln_coeff_ext[idof] = val;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices_int[idof]];
        coords_coeff_int[idof] = val;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices_ext[idof]];
        coords_coeff_ext[idof] = val;
    }

    std::vector<double> dual_int(n_soln_dofs_int);
    std::vector<double> dual_ext(n_soln_dofs_ext);

    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_int[itest];
        dual_int[itest] = this->dual[global_residual_row];
    }
    for (unsigned int itest=0; itest<n_soln_dofs_ext; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_ext[itest];
        dual_ext[itest] = this->dual[global_residual_row];
    }

    std::vector<real> rhs_int(n_soln_dofs_int);
    std::vector<real> rhs_ext(n_soln_dofs_ext);
    real dual_dot_residual;

    assemble_face_term(
        cell,
        current_cell_index,
        neighbor_cell_index,
        soln_coeff_int,
        soln_coeff_ext,
        coords_coeff_int,
        coords_coeff_ext,
        dual_int,
        dual_ext,
        face_subface_int,
        face_subface_ext,
        face_data_set_int,
        face_data_set_ext,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_int,
        fe_values_ext,
        penalty,
        fe_int,
        fe_ext,
        fe_metric,
        face_quadrature,
        rhs_int,
        rhs_ext,
        dual_dot_residual,
        compute_dRdW, compute_dRdX, compute_d2R);

    for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
        local_rhs_int_cell[itest_int] += getValue<real>(rhs_int[itest_int]);
    }
    for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
        local_rhs_ext_cell[itest_ext] += getValue<real>(rhs_ext[itest_ext]);
    }

}
#endif

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
void DGWeak<dim,nstate,real,MeshType>::assemble_volume_term(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const std::vector<real2> &soln_coeff, const std::vector<real2> &coords_coeff, const std::vector<real> &local_dual,
    const dealii::FESystem<dim,dim> &fe_soln, const dealii::FESystem<dim,dim> &fe_metric,
    const dealii::Quadrature<dim> &quadrature,
    const Physics::PhysicsBase<dim, nstate, real2> &physics,
    std::vector<real2> &rhs, real2 &dual_dot_residual,
    const bool compute_metric_derivatives,
    const dealii::FEValues<dim,dim> &fe_values_vol)
{
    (void) current_cell_index;
    using Array = std::array<real2, nstate>;
    using Tensor1D = dealii::Tensor<1,dim,real2>;
    using Tensor2D = dealii::Tensor<2,dim,real2>;
    using ArrayTensor = std::array<Tensor1D, nstate>;

    const unsigned int n_quad_pts      = quadrature.size();
    const unsigned int n_soln_dofs     = fe_soln.dofs_per_cell;

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        rhs[itest] = 0;
    }
    dual_dot_residual = 0.0;

    const std::vector<dealii::Point<dim>> &points = quadrature.get_points ();

    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    // Evaluate metric terms
    std::vector<Tensor2D> metric_jacobian;
    if (compute_metric_derivatives) metric_jacobian = evaluate_metric_jacobian ( points, coords_coeff, fe_metric);
    std::vector<real2> jac_det(n_quad_pts);
    std::vector<Tensor2D> jac_inv_tran(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        if (compute_metric_derivatives) {
            const real2 jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
            jac_det[iquad] = jacobian_determinant;

            const Tensor2D jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));
            jac_inv_tran[iquad] = jacobian_transpose_inverse;
        } else {
            jac_det[iquad] = fe_values_vol.JxW(iquad) / quadrature.weight(iquad);
        }
    }
#ifdef KOPRIVA_METRICS_VOL
    auto old_jac_inv_tran = jac_inv_tran;
    auto old_jac_det = jac_det;
    if constexpr (dim != 1) {
        evaluate_covariant_metric_jacobian<dim,real2> ( quadrature, coords_coeff, fe_metric, jac_inv_tran, jac_det);
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        if (abs(old_jac_det[iquad] - jac_det[iquad])/abs(old_jac_det[iquad]) > 1e-10) {
            std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
            std::cout << "Not the same jac det, iquad " << iquad << std::endl;
            std::cout << old_jac_det[iquad] << std::endl;
            std::cout << jac_det[iquad] << std::endl;
        }
        //for (int row=0;row<dim;++row) {
        //    for (int col=0;col<dim;++col) {
        //        if (abs(old_jac_inv_tran[iquad][row][col] - jac_inv_tran[iquad][row][col]) > 1e-10) {
        //            std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
        //            std::cout << "Not the same jac inv tran, iquad " << iquad << " row " << row << " col " << col << std::endl;
        //            std::cout << old_jac_inv_tran[iquad][row][col] << std::endl;
        //            std::cout << jac_inv_tran[iquad][row][col] << std::endl;
        //        }
        //    }
        //}
        //real2 mnorm = 0;
        //for (int row=0;row<dim;++row) {
        //    for (int col=0;col<dim;++col) {
        //        real2 val = (old_jac_inv_tran[iquad][row][col] - jac_inv_tran[iquad][row][col]);
        //        mnorm += val*val;
        //    }
        //}
        //mnorm = sqrt(mnorm);
        //if (mnorm > 1e-10) {
        //    std::cout << "Not the same jac inv tran, iquad " << iquad << std::endl;
        //    for (int row=0;row<dim;++row) {
        //        for (int col=0;col<dim;++col) {
        //            std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
        //            std::cout << old_jac_inv_tran[iquad][row][col] << " ";
        //        }
        //        std::cout << std::endl;
        //    }
        //    std::cout << std::endl;
        //    for (int row=0;row<dim;++row) {
        //        for (int col=0;col<dim;++col) {
        //            std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
        //            std::cout << jac_inv_tran[iquad][row][col] << " ";
        //        }
        //        std::cout << std::endl;
        //    }
        //}
    }
#endif


    // Build operators.
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = quadrature.get_points();
    dealii::FullMatrix<real> interpolation_operator(n_soln_dofs,n_quad_pts);
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            interpolation_operator[idof][iquad] = fe_soln.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    // Might want to have the dimension as the innermost index
    // Need a contiguous 2d-array structure
    // std::array<dealii::FullMatrix<real2>,dim> gradient_operator;
    // for (int d=0;d<dim;++d) {
    //     gradient_operator[d].reinit(n_soln_dofs, n_quad_pts);
    // }
    std::array<dealii::FullMatrix<real2>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(dealii::TableIndices<2>(n_soln_dofs, n_quad_pts));
    }
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
         for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
             if (compute_metric_derivatives) {
                 //const dealii::Tensor<1,dim,real2> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran[iquad], fe_soln.shape_grad(idof,points[iquad]));
                 const dealii::Tensor<1,dim,real2> ref_shape_grad = fe_soln.shape_grad(idof,points[iquad]);
                 dealii::Tensor<1,dim,real2> phys_shape_grad;
                 for (int dr=0;dr<dim;++dr) {
                     phys_shape_grad[dr] = 0.0;
                     for (int dc=0;dc<dim;++dc) {
                         phys_shape_grad[dr] += jac_inv_tran[iquad][dr][dc] * ref_shape_grad[dc];
                     }
                 }
                 for (int d=0;d<dim;++d) {
                     gradient_operator[d][idof][iquad] = phys_shape_grad[d];
                 }

                 // Exact mapping
                 // for (int d=0;d<dim;++d) {
                 //     const unsigned int istate = fe_soln.system_to_component_index(idof).first;
                 //     gradient_operator[d][idof][iquad] = fe_values_vol.shape_grad_component(idof, iquad, istate)[d];
                 // }
             } else {
                 for (int d=0;d<dim;++d) {
                     const unsigned int istate = fe_soln.system_to_component_index(idof).first;
                     gradient_operator[d][idof][iquad] = fe_values_vol.shape_grad_component(idof, iquad, istate)[d];
                 }
             }
         }
     }



    //const real2 cell_diameter = fe_values_.get_cell()->diameter();
    // real2 cell_volume = 0.0;
    // for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

    //     const real2 JxW_iquad = jac_det[iquad] * quadrature.weight(iquad);

    //     cell_volume = cell_volume + JxW_iquad;
    // }
    //const real2 cell_diameter = pow(cell_volume,1.0/dim);
    //const real2 artificial_diss_coeff = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
    //                                    this->discontinuity_sensor(cell_diameter, soln_coeff, fe_soln)
    //                                    : 0.0;
    const real2 artificial_diss_coeff = this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ?
                                        this->artificial_dissipation_coeffs[current_cell_index]
                                        : 0.0;
    (void) artificial_diss_coeff;

    typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
        this->triangulation.get(), cell->level(), cell->index(), &(this->dof_handler_artificial_dissipation));
    const unsigned int n_dofs_arti_diss = this->fe_q_artificial_dissipation.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);
    artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);
/*
    std::vector<real> artificial_diss_coeff_at_q(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        artificial_diss_coeff_at_q[iquad] = 0.0;

        if ( this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ) {
            const dealii::Point<dim,real> point = unit_quad_pts[iquad];
            for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
                const unsigned int index = dof_indices_artificial_dissipation[idof];
                artificial_diss_coeff_at_q[iquad] += this->artificial_dissipation_c0[index] * this->fe_q_artificial_dissipation.shape_value(idof, point);
            }
        }
    }

*/

    std::vector<real2> artificial_diss_coeff_at_q(n_quad_pts);
    real2 arti_diss = this->discontinuity_sensor(quadrature, soln_coeff, fe_soln, jac_det);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad)
    {
        artificial_diss_coeff_at_q[iquad] = arti_diss;
       /* dealii::Point<dim,real> point = unit_quad_pts[iquad];
        // Rescale over -1,1
        for (int d=0; d<dim; ++d)
        {
            point[d] = point[d]*2 - 1.0;
        }
        double gegenbauer_factor = 0.1;
        double gegenbauer = 1.0;
        for (int d=0; d<dim; ++d)
        {
            gegenbauer *= std::pow(1-point[d]*point[d], gegenbauer_factor);
        }
        artificial_diss_coeff_at_q[iquad] = arti_diss * gegenbauer;*/
    }

    std::vector< Array > soln_at_q(n_quad_pts);
    std::vector< ArrayTensor > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

    std::vector< ArrayTensor > conv_phys_flux_at_q(n_quad_pts);
    std::vector< ArrayTensor > diss_phys_flux_at_q(n_quad_pts);
    std::vector< Array > source_at_q(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) {
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
            const unsigned int istate = fe_soln.system_to_component_index(idof).first;
            soln_at_q[iquad][istate]      += soln_coeff[idof] * interpolation_operator[idof][iquad];
            for (int d=0;d<dim;++d) {
                soln_grad_at_q[iquad][istate][d] += soln_coeff[idof] * gradient_operator[d][idof][iquad];
            }
        }
        conv_phys_flux_at_q[iquad] = physics.convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = physics.dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);

        if (this->all_parameters->artificial_dissipation_param.add_artificial_dissipation) {
            ArrayTensor artificial_diss_phys_flux_at_q;
            //artificial_diss_phys_flux_at_q = physics.artificial_dissipative_flux (artificial_diss_coeff, soln_at_q[iquad], soln_grad_at_q[iquad]);
            artificial_diss_phys_flux_at_q = DGBaseState<dim,nstate,real,MeshType>::artificial_dissip->calc_artificial_dissipation_flux(soln_at_q[iquad], soln_grad_at_q[iquad],artificial_diss_coeff_at_q[iquad]);
            for (int s=0; s<nstate; s++) {
                diss_phys_flux_at_q[iquad][s] += artificial_diss_phys_flux_at_q[s];
            }
        }

        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            dealii::Point<dim,real2> ad_point;
            for (int d=0;d<dim;++d) { ad_point[d] = 0.0;}
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const int iaxis = fe_metric.system_to_component_index(idof).first;
                ad_point[iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
            }
            source_at_q[iquad] = physics.source_term (ad_point, soln_at_q[iquad], DGBase<dim,real,MeshType>::current_time);
            //Array artificial_source_at_q = physics.artificial_source_term (artificial_diss_coeff, ad_point, soln_at_q[iquad]);
            //Array artificial_source_at_q = physics.artificial_source_term (artificial_diss_coeff_at_q[iquad], ad_point, soln_at_q[iquad]);
            //for (int s=0;s<nstate;++s) source_at_q[iquad][s] += artificial_source_at_q[s];
        }
    }

    // Weak form
    // The right-hand side sends all the term to the side of the source term
    // Therefore,
    // \divergence ( Fconv + Fdiss ) = source
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

        const unsigned int istate = fe_soln.system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const real2 JxW_iquad = jac_det[iquad] * quadrature.weight(iquad);

            for (int d=0;d<dim;++d) {
                // Convective
                rhs[itest] = rhs[itest] + gradient_operator[d][itest][iquad] * conv_phys_flux_at_q[iquad][istate][d] * JxW_iquad;
                //// Diffusive
                //// Note that for diffusion, the negative is defined in the physics
                rhs[itest] = rhs[itest] + gradient_operator[d][itest][iquad] * diss_phys_flux_at_q[iquad][istate][d] * JxW_iquad;
            }
            // Source
            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
                rhs[itest] = rhs[itest] + interpolation_operator[itest][iquad]* source_at_q[iquad][istate] * JxW_iquad;
            }
        }

        dual_dot_residual += local_dual[itest]*rhs[itest];

    }

}

#ifdef FADFAD
template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_volume_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;

    using ADArray = std::array<FadFadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadFadType>, nstate >;

    const unsigned int n_soln_dofs     = fe_soln.dofs_per_cell;

    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    std::vector<FadFadType> coords_coeff(n_metric_dofs);
    std::vector<FadFadType> soln_coeff(n_soln_dofs);

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices[itest];
        local_dual[itest] = this->dual[global_residual_row];
    }

    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    const bool compute_metric_derivatives = true;//(!compute_dRdX && !compute_d2R) ? false : true;

    unsigned int w_start, w_end, x_start, x_end;
    automatic_differentiation_indexing_1( compute_dRdW, compute_dRdX, compute_d2R,
                                          n_soln_dofs, n_metric_dofs,
                                          w_start, w_end, x_start, x_end );

    unsigned int i_derivative = 0;
    const unsigned int n_total_indep = x_end;

    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;
        soln_coeff[idof].val() = val;

        if (compute_dRdW || compute_d2R) soln_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) soln_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdW || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;
        coords_coeff[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdX || compute_d2R) i_derivative++;
    }

    AssertDimension(i_derivative, n_total_indep);

    FadFadType dual_dot_residual = 0.0;
    std::vector<FadFadType> rhs(n_soln_dofs);
    assemble_volume_term<FadFadType>(
        cell,
        current_cell_index,
        soln_coeff, coords_coeff, local_dual,
        fe_soln, fe_metric, quadrature,
        *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_fad_fad),
        rhs, dual_dot_residual,
        compute_metric_derivatives, fe_values_vol);

    // Weak form
    // The right-hand side sends all the term to the side of the source term
    // Therefore,
    // \divergence ( Fconv + Fdiss ) = source
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

        if (compute_dRdW) {
            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = rhs[itest].dx(i_dx).val();
                AssertIsFinite(residual_derivatives[idof]);
            }
            const bool elide_zero_values = false;
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives, elide_zero_values);
        }
        if (compute_dRdX) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = rhs[itest].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }
        //if (compute_d2R) {
        //    const unsigned int global_residual_row = soln_dof_indices[itest];
        //    dual_dot_residual += this->dual[global_residual_row]*rhs[itest];
        //}

        local_rhs_cell(itest) += rhs[itest].val().val();
        AssertIsFinite(local_rhs_cell(itest));

    }


    if (compute_d2R) {

        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);

        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;
            const FadType dWi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_soln_dofs; ++jdof) {
                const unsigned int j_dx = jdof+w_start;
                dWidW[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices[idof], soln_dof_indices, dWidW);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices[idof], metric_dof_indices, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_start;
            const FadType dXi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = dXi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }
    }

}
#endif
#ifndef FADFAD
template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
void DGWeak<dim,nstate,real,MeshType>::assemble_volume_codi_taped_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const dealii::FEValues<dim,dim> &fe_values_lagrange,
    const Physics::PhysicsBase<dim, nstate, real2> &physics,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) fe_values_lagrange;

    using adtype = real2;

    using ADArray = std::array<adtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,adtype>, nstate >;

    const unsigned int n_soln_dofs = fe_soln.dofs_per_cell;

    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    std::vector<adtype> coords_coeff(n_metric_dofs);
    std::vector<adtype> soln_coeff(n_soln_dofs);

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices[itest];
        local_dual[itest] = this->dual[global_residual_row];
    }

    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    const bool compute_metric_derivatives = true;//(!compute_dRdX && !compute_d2R) ? false : true;

    unsigned int w_start, w_end, x_start, x_end;
    automatic_differentiation_indexing_1( compute_dRdW, compute_dRdX, compute_d2R,
                                          n_soln_dofs, n_metric_dofs,
                                          w_start, w_end, x_start, x_end );

    using TH = codi::TapeHelper<adtype>;
    TH th;
    adtype::getGlobalTape();
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.startRecording();
    }
    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;

        if (compute_dRdW || compute_d2R) {
            th.registerInput(soln_coeff[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(soln_coeff[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;

        if (compute_dRdX || compute_d2R) {
            th.registerInput(coords_coeff[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(coords_coeff[idof]);
        }
    }

    adtype dual_dot_residual = 0.0;
    std::vector<adtype> rhs(n_soln_dofs);
    assemble_volume_term<adtype>(
        cell,
        current_cell_index,
        soln_coeff, coords_coeff, local_dual,
        fe_soln, fe_metric, quadrature,
        physics,
        rhs, dual_dot_residual,
        compute_metric_derivatives, fe_values_vol);

    if (compute_dRdW || compute_dRdX) {
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
            th.registerOutput(rhs[itest]);
        }
    } else if (compute_d2R) {
        th.registerOutput(dual_dot_residual);
    }
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.stopRecording();
    }

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_rhs_cell(itest) += getValue<adtype>(rhs[itest]);
        AssertIsFinite(local_rhs_cell(itest));
    }

    if (compute_dRdW) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = jac(itest,i_dx);
                AssertIsFinite(residual_derivatives[idof]);
            }
            const bool elide_zero_values = false;
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives, elide_zero_values);
        }
        th.deleteJacobian(jac);

    }

    if (compute_dRdX) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = jac(itest,i_dx);
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }
        th.deleteJacobian(jac);
    }


    if (compute_d2R) {
        typename TH::HessianType& hes = th.createHessian();
        th.evalHessian(hes);

        int i_dependent = (compute_dRdW || compute_dRdX) ? n_soln_dofs : 0;

        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);

        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;

            for (unsigned int jdof=0; jdof<n_soln_dofs; ++jdof) {
                const unsigned int j_dx = jdof+w_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices[idof], soln_dof_indices, dWidW);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices[idof], metric_dof_indices, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_start;

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }

        th.deleteHessian(hes);
    }

    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        adtype::getGlobalTape().deactivateValue(soln_coeff[idof]);
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        adtype::getGlobalTape().deactivateValue(coords_coeff[idof]);
    }

}

template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_volume_residual(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const dealii::FEValues<dim,dim> &fe_values_lagrange,
    const Physics::PhysicsBase<dim, nstate, real> &physics,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) fe_values_lagrange;
    (void) compute_dRdW;
    (void) compute_dRdX;
    (void) compute_d2R;
    assert( !compute_dRdW && !compute_dRdX && !compute_d2R);
    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    const bool compute_metric_derivatives = true;//(!compute_dRdX && !compute_d2R) ? false : true;

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_soln_dofs = fe_soln.dofs_per_cell;

    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    std::vector<double> coords_coeff(n_metric_dofs);
    std::vector<double> soln_coeff(n_soln_dofs);

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices[itest];
        local_dual[itest] = this->dual[global_residual_row];
    }

    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid->volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;
    }

    double dual_dot_residual = 0.0;
    std::vector<double> rhs(n_soln_dofs);
    assemble_volume_term<double>(
        cell,
        current_cell_index,
        soln_coeff, coords_coeff, local_dual,
        fe_soln, fe_metric, quadrature,
        physics,
        rhs, dual_dot_residual,
        compute_metric_derivatives, fe_values_vol);

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_rhs_cell(itest) += getValue<double>(rhs[itest]);
        AssertIsFinite(local_rhs_cell(itest));
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_volume_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const dealii::FEValues<dim,dim> &fe_values_lagrange,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;
    (void) fe_values_lagrange;
    if (compute_d2R) {
        assemble_volume_codi_taped_derivatives<codi_HessianComputationType>(
        cell,
            current_cell_index,
            fe_values_vol,
            fe_soln, quadrature,
            metric_dof_indices, soln_dof_indices,
            local_rhs_cell,
            fe_values_lagrange,
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad_fad),
            compute_dRdW, compute_dRdX, compute_d2R);
    } else if (compute_dRdW || compute_dRdX) {
        assemble_volume_codi_taped_derivatives<codi_JacobianComputationType>(
        cell,
            current_cell_index,
            fe_values_vol,
            fe_soln, quadrature,
            metric_dof_indices, soln_dof_indices,
            local_rhs_cell,
            fe_values_lagrange,
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad),
            compute_dRdW, compute_dRdX, compute_d2R);
    } else {
        assemble_volume_residual(
        cell,
            current_cell_index,
            fe_values_vol,
            fe_soln, quadrature,
            metric_dof_indices, soln_dof_indices,
            local_rhs_cell,
            fe_values_lagrange,
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_double),
            compute_dRdW, compute_dRdX, compute_d2R);
    }

    return;
}

template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_face_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const std::pair<unsigned int, int> face_subface_int,
    const std::pair<unsigned int, int> face_subface_ext,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
    const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_int,
    const dealii::FESystem<dim,dim> &fe_ext,
    const dealii::Quadrature<dim-1> &face_quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;
    if (compute_d2R) {
        assemble_face_codi_taped_derivatives<codi_HessianComputationType>(
        cell,
            current_cell_index,
            neighbor_cell_index,
            face_subface_int,
            face_subface_ext,
            face_data_set_int,
            face_data_set_ext,
            fe_values_int,
            fe_values_ext,
            penalty,
            fe_int,
            fe_ext,
            face_quadrature,
            metric_dof_indices_int,
            metric_dof_indices_ext,
            soln_dof_indices_int,
            soln_dof_indices_ext,
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad_fad),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_rad_fad),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_rad_fad),
            local_rhs_int_cell,
            local_rhs_ext_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    } else if (compute_dRdW || compute_dRdX) {
        assemble_face_codi_taped_derivatives<codi_JacobianComputationType>(
        cell,
            current_cell_index,
            neighbor_cell_index,
            face_subface_int,
            face_subface_ext,
            face_data_set_int,
            face_data_set_ext,
            fe_values_int,
            fe_values_ext,
            penalty,
            fe_int,
            fe_ext,
            face_quadrature,
            metric_dof_indices_int,
            metric_dof_indices_ext,
            soln_dof_indices_int,
            soln_dof_indices_ext,
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_rad),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_rad),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_rad),
            local_rhs_int_cell,
            local_rhs_ext_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    } else {
        assemble_face_residual(
        cell,
            current_cell_index,
            neighbor_cell_index,
            face_subface_int,
            face_subface_ext,
            face_data_set_int,
            face_data_set_ext,
            fe_values_int,
            fe_values_ext,
            penalty,
            fe_int,
            fe_ext,
            face_quadrature,
            metric_dof_indices_int,
            metric_dof_indices_ext,
            soln_dof_indices_int,
            soln_dof_indices_ext,
            *(DGBaseState<dim,nstate,real,MeshType>::pde_physics_double),
            *(DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_double),
            *(DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double),
            local_rhs_int_cell,
            local_rhs_ext_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    }

    return;
}
#endif


template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::allocate_auxiliary_equation ()
{
    //Do Nothing.
}
template <int dim, int nstate, typename real, typename MeshType>
void DGWeak<dim,nstate,real,MeshType>::assemble_auxiliary_residual ()
{
    //Do Nothing.
}


// using default MeshType = Triangulation
// 1D: dealii::Triangulation<dim>;
// OW: dealii::parallel::distributed::Triangulation<dim>;
template class DGWeak <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class DGWeak <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class DGWeak <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGWeak <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace

