#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/grid/grid_tools.h>

#include "reconstruct_poly.h"

namespace PHiLiP {

namespace GridRefinement {

// takes an input field and polynomial space and output the largest directional derivative and coresponding normal direction
template <int dim, typename real>
void ReconstructPoly<dim,real>::reconstruct_directional_derivative(
    dealii::LinearAlgebra::distributed::Vector<real>&solution,              // approximation to be reconstructed
    dealii::hp::DoFHandler<dim>&                     dof_handler,           // dof_handler
    dealii::hp::MappingCollection<dim>&              mapping_collection,    // mapping collection
    dealii::hp::FECollection<dim>&                   fe_collection,         // fe collection
    dealii::hp::QCollection<dim>&                    quadrature_collection, // quadrature collection
    dealii::UpdateFlags&                             update_flags,          // update flags for for volume fe
    unsigned int                                     rel_order,             // order of the reconstruction
    dealii::Vector<dealii::Tensor<1,dim,real>>&      A)                     // (output) holds the largest (scaled) derivative in each direction and then in each orthogonal plane
{
    const real pi = atan(1)*4.0;

    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        // generating the polynomial space
        unsigned int order = cell->active_fe_index()+rel_order;
        dealii::PolynomialSpace<dim> poly_space(dealii::Polynomials::Monomial<double>::generate_complete_basis(order));


        // getting the vector of polynomial coefficients from the p+1 expansion
        dealii::Vector<real> coeffs_non_hom = reconstruct_H1_norm(
            cell,
            solution,
            poly_space,
            mapping_collection,
            fe_collection,
            quadrature_collection,
            update_flags);

        const unsigned int n_poly = poly_space.n();

        // assembling a vector of coefficients and indices
        std::vector<real>                          coeffs;
        std::vector<std::array<unsigned int, dim>> indices;
        unsigned int                               n_vec = 0;

        for(unsigned int i = 0; i < n_poly; ++i){
            std::array<unsigned int, dim> arr = compute_index(i, order);

            unsigned int sum = 0;
            for(int j = 0; j < dim; ++j)
                sum += arr[j];

            if(sum == order){
                // based on expansion of taylor series additional term from expansion (x (+y) (+z))^n
                // for cross terms, in 1D no such terms. But, after expanding the n^th derivative with
                // i, j partials in (x,y) we get 1/n! * (n \choose i, j) * i! j! = 1 on the x^i y^j term
                // (the only one that will be remaining). Also generalizes to n-dimensions.
                coeffs.pushback(coeffs_non_hom[i]);
                indices.pushback(arr);
                n_vec++;
            }
        }

        dealii::Tensor<1,dim,real> A_cell;
        if(dim == 1){
            Assert(n_vec == 1, dealii::ExcInternalError());

            A_cell[0] = coeffs[0];
        }else if(order == 2){
            // if current order is 2, can be solved by the eigenvalue problem
            Assert(n_vec == dim*(dim+1)/2, dealii::ExcInternalError());

            dealii::SymmetricTensor<2,dim,real> hessian;
            for(unsigned int n = 0; n < n_vec; ++n)
                for(unsigned int i = 0; i < dim; ++i)
                    for(unsigned int j = i+1; j < dim; ++j)
                        if(indices[n][i] && indices[n][j])
                            hessian[i][j] = coeffs[n] * (i==j ? 1.0 : 0.5);

            // see also dealii::eigenvectors(SymmetricTensor T)
            // https://www.dealii.org/current/doxygen/deal.II/symmetric__tensor_8h.html#a45c9cd0a3fecbd58ae133dfdd104f9f9
            //std::array< Number, 3 >
            std::array<real,dim> eig = dealii::eigenvalues<real,dim>(hessian);
            
            for(int d = 0; d < dim; ++d)
                A_cell[d] = eig[d];
        }else{
            // evaluating any point requires sum over power of the multindices
            auto eval = [&](dealii::Tensor<1,dim,real>& point) -> real{
                real val = 0.0;
                for(int i = 0; i < n_vec; ++i){
                    real val_coeff = coeffs[i];
                    for(int d = 0; d < dim; ++d)
                        val_coeff *= pow(point[d], indices[i][d]);
                    val += val_coeff;
                }
                return val;
            };

            // looping over the range
            if(dim == 2){
                // number of sampling points in each direciton
                const unsigned int n_sample = 180;

                // keeping track of largest point and angle
                real A_max = 0.0, t_max = 0.0;

                // using polar coordinates theta\in[0, \pi)
                real r = 1.0, theta, val;
                for(unsigned int i = 0; i < n_sample; ++i){
                    theta = i*pi/n_sample;
                    dealii::Tensor<1,dim,real> p(r*cos(theta), r*sin(theta));
                    
                    val = abs(eval(p));
                    if(val > A_max){
                        A_max = val;
                        t_max = theta;
                    }
                }

                // Taking A_2 to be at an angle of 90 degrees relative to first
                dealii::Tensor<1,dim,real> p(r*cos(t_max+pi/2.0), r*sin(t_max+pi/2.0));
                
                A_cell[0] = A_max;
                A_cell[1] = abs(eval(p));
            }else if(dim == 3){
                // using fibbonaci sphere algorithm, with ~ n^2/2 points compared to 2d for equal points in theta and phi as before
                // https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012
                const unsigned int n_sample = 180, n_sample_3d = 180*90;

                // keeping track of the largest point and angles
                real A_1 = 0.0;
                dealii::Tensor<1,dim,real> p_1;

                // parameters needed
                real offset    = 1.0/n_sample_3d;
                real increment = pi * (3 - sqrt(5));

                // spherical coordinates
                real y, r, phi, x, z, val;
                for(unsigned int i = 0; i < n_sample_3d; ++i){
                    // calculation of the points 
                    y = (i*offset) - 1 + offset/2;
                    r = sqrt(1-pow(y,2));

                    phi = remainder(i, 2*n_sample_3d) * increment;

                    dealii::Tensor<1,dim,real> p(r*cos(phi),y,r*sin(phi));
                    val = abs(eval(p));
                    if(val > A_1){
                        A_1 = val;
                        p_1 = p/p.norm();
                    }
                }

                // generating the rest of the basis for p_1 rotation, two orthogonal vectors forming a plane
                dealii::Tensor<1,dim,real> u;
                dealii::Tensor<1,dim,real> v;

                // checking if near the x-axis
                dealii::Tensor<1,dim,real> px();
                px[0] = 1.0;
                if(abs(px*p_1) < 1.0/sqrt(2.0)){ // if further apart than 45 degrees use x-axis
                    // using cross products to generate two vectors orthogonal to p_1
                    u = cross_product_3d(p_1, px);
                }else{ // if not, use the y axis instead
                    dealii::Tensor<1,dim,real> py();
                    py[1] = 1.0;
                    u = cross_product_3d(p_1, py);
                }
                // second orthogonal to form the basis
                v = cross_product_3d(p_1, u);
                
                // now performing the 2d analysis in the plane uv
                real A_2 = 0.0, t_2 = 0.0;

                // using polar coordinates theta\in[0, \pi)
                real theta;
                for(unsigned int i = 0; i < n_sample; ++i){
                    theta = i*pi/n_sample;
                    dealii::Tensor<1,dim,real> p_2 = r*cos(theta)*u + r*sin(theta)*v;
                    
                    val = abs(eval(p_2));
                    if(val > A_2){
                        A_2 = val;
                        t_2 = theta;
                    }
                }

                // Taking A_2 to be at an angle of 90 degrees relative to first
                dealii::Tensor<1,dim,real> p_3 = r*cos(t_2+pi/2.0)*u + r*sin(t_2+pi/2.0)*v;
                
                // assigning the results
                A_cell[0] = A_1;
                A_cell[1] = A_2;
                A_cell[2] = abs(eval(p_3));
            }else{ // no other dimensions should appear
                Assert(false, dealii::ExcInternalError());
            }
        }

        // storing the tensor of results
        A[cell->active_cell_index()] = A_cell;
    }
}

// from DEALII, https://www.dealii.org/current/doxygen/deal.II/polynomial__space_8cc_source.html
// protected function so reimplementing slightly modified form here
// computes the multiindex for different dimensions, assuming the index map hasn't been modified
template <>
std::array<unsigned int, 1> ReconstructPoly<1,double>::compute_index(
    const unsigned int i,
    const unsigned int /*size*/)
{
    return {{i}};
}

template <>
std::array<unsigned int, 2> ReconstructPoly<2,double>::compute_index(
    const unsigned int i,
    const unsigned int size)
{
    unsigned int k = 0;
    for(unsigned int iy = 0; iy < size; ++iy)
        if(i < k+size-iy){
            return {{i-k, iy}};
        }else{
            k += size - iy;
        }

    Assert(false, dealii::ExcInternalError());
    return {{dealii::numbers::invalid_unsigned_int, dealii::numbers::invalid_unsigned_int}};
}

template <>
std::array<unsigned int, 3> ReconstructPoly<3,double>::compute_index(
    const unsigned int i,
    const unsigned int size)
{
    unsigned int k = 0;
    for(unsigned int iz = 0; iz < size; ++iz)
        for(unsigned int iy = 0; iy < size - iz; ++iy)
            if(i < k+size-iy){
                return {{i-k, iy, iz}};
            }else{
                k += size - iy - iz;
            }

    Assert(false, dealii::ExcInternalError());
    return {{dealii::numbers::invalid_unsigned_int, dealii::numbers::invalid_unsigned_int, dealii::numbers::invalid_unsigned_int}};
}

// TODO: add nstate template or move to grid_refinement
// check that the vectors are properly updated for ghosted dofs
// trying to pass everything by ref, may run into trouble later
// template <int dim, int nstate, typename real>
template <int dim, typename real>
template <typename DoFCellAccessorType>
dealii::Vector<real> ReconstructPoly<dim,real>::reconstruct_H1_norm(
    DoFCellAccessorType &                             curr_cell,
    dealii::PolynomialSpace<dim>                      ps,
    dealii::LinearAlgebra::distributed::Vector<real> &solution,
    unsigned int                                      order,
    dealii::hp::MappingCollection<dim> &              mapping_collection,
    dealii::hp::FECollection<dim> &                   fe_collection,
    dealii::hp::QCollection<dim> &                    quadrature_collection,
    dealii::UpdateFlags &                             update_flags)
{
    const int nstate = 1;

    // center point of the current cell
    dealii::Point<dim,real> center_point = curr_cell->center();

    // things to be extracted
    std::vector<std::array<real,nstate>>                  soln_at_q_vec;
    std::vector<std::array<dealii::Tensor<1,dim>,nstate>> grad_at_q_vec; // for the H^1 norm
    std::vector<dealii::Point<dim,real>>                  qpoint_vec;
    std::vector<real>                                     JxW_vec;

    // and keeping track of vector lengths
    int n_vec = 0;

    // fe_values
    dealii::hp::FEValues<dim,dim> fe_values_collection(
        mapping_collection,
        fe_collection,
        quadrature_collection,
        update_flags);

    // looping over the cell vector and extracting the soln, qpoint and JxW
    std::vector<DoFCellAccessorType> cell_patch = dealii::GridTools::get_patch_around_cell(curr_cell);
    for(auto cell : cell_patch){
        const unsigned int mapping_index = 0;
        const unsigned int fe_index = cell->active_fe_index();
        const unsigned int quad_index = fe_index; 

        const unsigned int n_dofs = fe_collection[fe_index].n_dofs_per_cell();
        const unsigned int n_quad = quadrature_collection[quad_index].size();

        fe_values_collection.reinit(cell, quad_index, mapping_index, fe_index);
        const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();

        std::vector<dealii::types::global_dof_index> dofs_indices(fe_values.dofs_per_cell);
        cell->get_dof_indices(dofs_indices);

        // looping over the quadrature points of this cell
        std::array<real,nstate>                  soln_at_q;
        std::array<dealii::Tensor<1,dim>,nstate> grad_at_q;
        for(unsigned int iquad = 0; iquad < n_quad; ++iquad){
            soln_at_q.fill(0.0);
            grad_at_q.fill(0.0); // Tensor = 0 should asign 0 to all components

            // looping over the DoFS to get the solution value
            for(unsigned int idof = 0; idof < n_dofs; ++idof){
                const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += solution[dofs_indices[idof]] * fe_values.shape_value_component(idof, iquad, istate);
                grad_at_q[istate] += solution[dofs_indices[idof]] * fe_values.shape_grad_component(idof, iquad, istate);
            }

            // push back into the vectors
            soln_at_q_vec.push_back(soln_at_q);
            grad_at_q_vec.push_back(grad_at_q);
            qpoint_vec.push_back(fe_values.quadrature_point(iquad) - center_point);
            JxW_vec.push_back(fe_values.JxW(iquad));
            n_vec++;
        }
    }

    // number of polynomials in the space
    const unsigned int n_poly = ps.n();

    // allocating the matrix and vector for the RHS
    dealii::FullMatrix<real> mat(n_poly);
    dealii::Vector<real>     rhs(n_poly);

    // looping over to assemble the matrices
    mat.fill(0.0);
    for(unsigned int i_poly = 0; i_poly < n_poly; ++i_poly){
        for(unsigned int j_poly = 0; j_poly < n_poly; ++j_poly){
            // taking the inner product between \psi_i and \psi_j
            // <u,v>_{H^1(\Omega)} = \int_{\Omega} u*v + \sum_i^N {\partial_i u * \partial_i v} dx
            for(unsigned int i_vec = 0; i_vec < n_vec; ++i_vec)
                mat.add(i_poly, j_poly, 
                    ((ps.compute_value(i_poly, qpoint_vec[i_vec]) * ps.compute_value(j_poly, qpoint_vec[i_vec]))
                    +(ps.compute_grad(i_poly, qpoint_vec[i_vec])  * ps.compute_grad(j_poly, qpoint_vec[i_vec])))
                    *JxW_vec[i_vec]);
        }

        // take inner product of \psi_i and u (soluition)
        for(unsigned int i_vec = 0; i_vec < n_vec; ++i_vec)
            rhs[i_poly] += ((ps.compute_value(i_poly, qpoint_vec[i_vec]) * soln_at_q_vec[i_vec])
                           +(ps.compute_grad(i_poly, qpoint_vec[i_vec]) * grad_at_q_vec[i_vec]))
                           *JxW_vec[i_vec];
    }

    // solving the system
    dealii::Vector<real> coeffs(n_poly);

    mat.gauss_jordan();
    mat.vmult(coeffs, rhs);

    return coeffs;
}

} // namespace GridRefinement

} // namespace PHiLiP
