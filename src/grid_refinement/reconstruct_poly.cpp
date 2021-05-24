#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/grid/grid_tools.h>

#include "reconstruct_poly.h"
#include "physics/manufactured_solution.h"

namespace PHiLiP {

namespace GridRefinement {

template <int dim, int nstate, typename real>
ReconstructPoly<dim,nstate,real>::ReconstructPoly(
        const dealii::DoFHandler<dim>&            dof_handler,           // dof_handler
        const dealii::hp::MappingCollection<dim>& mapping_collection,    // mapping collection
        const dealii::hp::FECollection<dim>&      fe_collection,         // fe collection
        const dealii::hp::QCollection<dim>&       quadrature_collection, // quadrature collection
        const dealii::UpdateFlags&                update_flags) :        // update flags for for volume fe
            dof_handler(dof_handler),
            mapping_collection(mapping_collection),
            fe_collection(fe_collection),
            quadrature_collection(quadrature_collection),
            update_flags(update_flags),
            norm_type(NormType::H1)
{
    reinit(dof_handler.get_triangulation().n_active_cells());
}

template <int dim, int nstate, typename real>
void ReconstructPoly<dim,nstate,real>::set_norm_type(const NormType norm_type)
{
    this->norm_type = norm_type;
}

template <int dim, int nstate, typename real>
void ReconstructPoly<dim,nstate,real>::reinit(const unsigned int n)
{
    derivative_value.resize(n);
    derivative_direction.resize(n);
}

// reconstruct the directional derivatives of the reconstructed solution along each of the quad chords
template <int dim, int nstate, typename real>
void ReconstructPoly<dim,nstate,real>::reconstruct_chord_derivative(
    const dealii::LinearAlgebra::distributed::Vector<real>& solution,  // solution approximation to be reconstructed
    const unsigned int                                      rel_order) // order of the apporximation
{
    /* based on the dealii numbering, chords defined along nth axis
            ^ dir[1]
            |
        2---+---3
        |   |   |
        +---o---+---> dir[0]
        |   |   |
        0---+---1
    */

    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        // generating the polynomial space
        unsigned int order = cell->active_fe_index()+rel_order;
        dealii::PolynomialSpace<dim> poly_space(dealii::Polynomials::Monomial<double>::generate_complete_basis(order));

        // getting the vector of polynomial coefficients from the p+1 expansion
        dealii::Vector<real> coeffs_non_hom = reconstruct_norm(
            norm_type,
            cell,
            poly_space,
            solution);

        const unsigned int n_poly   = poly_space.n();
        const unsigned int n_degree = poly_space.degree();

        // assembling a vector of coefficients and indices
        std::vector<real>                          coeffs;
        std::vector<std::array<unsigned int, dim>> indices;
        unsigned int                               n_vec = 0;

        for(unsigned int i = 0; i < n_poly; ++i){
            std::array<unsigned int, dim> arr = compute_index<dim>(i, n_degree);

            unsigned int sum = 0;
            for(int j = 0; j < dim; ++j)
                sum += arr[j];

            if(sum == order){
                // based on expansion of taylor series additional term from expansion (x (+y) (+z))^n
                // for cross terms, in 1D no such terms. But, after expanding the n^th derivative with
                // i, j partials in (x,y) we get 1/n! * (n \choose i, j) * i! j! = 1 on the x^i y^j term
                // (the only one that will be remaining). Also generalizes to n-dimensions.
                coeffs.push_back(coeffs_non_hom[i]);
                indices.push_back(arr);
                n_vec++;
            }
        }

        std::array<real,dim> A_cell;
        std::array<dealii::Tensor<1,dim,real>,dim> chord_vec;

        // holds the nodes that form the chord
        // summing over all the nodes onto each (dim) neighbouring faces/edges
        std::array<std::pair<dealii::Tensor<1,dim,real>, dealii::Tensor<1,dim,real>>,dim> chord_nodes;
        for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex){
            
            // logic decides what side of each axis vertex is on
            for(unsigned int i = 0; i < dim; ++i){

                // chord side equivalent to sign of i^th bit in binary
                // uses bit-shift and remainder to determine if this bit is 0/1
                /* example for 2D (see figure above):
                    vertex  b0  b1 
                    0       0   0
                    1       1   0
                    2       0   1
                    3       1   1
                 */
                if(vertex>>i % 2 == 0){
                    chord_nodes[i].first  += cell->vertex(vertex);
                }else{
                    chord_nodes[i].second += cell->vertex(vertex);
                }

            }

        }

        // computing the direction, the chords could also be divided by 2^{i-1} first to get the actual physical coordinates
        for(unsigned int i = 0; i < dim; ++i)
            chord_vec[i] = chord_nodes[i].second - chord_nodes[i].first;

        // normalizing
        for(unsigned int i = 0; i < dim; ++i)
            chord_vec[i] /= chord_vec[i].norm();
    
        // computing the directional derivative along each vector
        for(unsigned int i = 0; i < dim; ++i){ // loop over the axes
            A_cell[i] = 0;
            for(unsigned int n = 0; n < n_vec; ++n){ // loop over the polynomials
                real poly_val = coeffs[n];
                
                for(unsigned int d = 0; d < dim; ++d) // loop over each poly term, ie x^i y^j z^k
                    poly_val *= pow(chord_vec[i][d], indices[n][d]);

                // adding polynomial terms contribution to the axis
                A_cell[i] += poly_val;
            }
        }

        const unsigned int index = cell->active_cell_index();
        derivative_value[index]     = A_cell;
        derivative_direction[index] = chord_vec;
    }

}

// takes an input field and polynomial space and output the largest directional derivative and coresponding normal direction
template <int dim, int nstate, typename real>
void ReconstructPoly<dim,nstate,real>::reconstruct_directional_derivative(
    const dealii::LinearAlgebra::distributed::Vector<real>&  solution,  // solution approximation to be reconstructed
    const unsigned int                                       rel_order) // order of the apporximation
{
    const real pi = atan(1)*4.0;

    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        // generating the polynomial space
        unsigned int order = cell->active_fe_index()+rel_order;
        dealii::PolynomialSpace<dim> poly_space(dealii::Polynomials::Monomial<double>::generate_complete_basis(order));

        // getting the vector of polynomial coefficients from the p+1 expansion
        dealii::Vector<real> coeffs_non_hom = reconstruct_norm(
            norm_type,
            cell,
            poly_space,
            solution);

        const unsigned int n_poly   = poly_space.n();
        const unsigned int n_degree = poly_space.degree();

        // assembling a vector of coefficients and indices
        std::vector<real>                          coeffs;
        std::vector<std::array<unsigned int, dim>> indices;
        unsigned int                               n_vec = 0;

        for(unsigned int i = 0; i < n_poly; ++i){
            std::array<unsigned int, dim> arr = compute_index<dim>(i, n_degree);

            unsigned int sum = 0;
            for(int j = 0; j < dim; ++j)
                sum += arr[j];

            if(sum == order){
                // based on expansion of taylor series additional term from expansion (x (+y) (+z))^n
                // for cross terms, in 1D no such terms. But, after expanding the n^th derivative with
                // i, j partials in (x,y) we get 1/n! * (n \choose i, j) * i! j! = 1 on the x^i y^j term
                // (the only one that will be remaining). Also generalizes to n-dimensions.
                coeffs.push_back(coeffs_non_hom[i]);
                indices.push_back(arr);
                n_vec++;
            }
        }

        std::array<real,dim>                       value_cell;
        std::array<dealii::Tensor<1,dim,real>,dim> direction_cell;
        if(dim == 1){

            Assert(n_vec == 1, dealii::ExcInternalError());

            value_cell[0]        = coeffs[0];
            direction_cell[0][0] = 1.0;

        }else if(order == 2){

            // if current order is 2, can be solved by the eigenvalue problem
            Assert(n_vec == dim*(dim+1)/2, dealii::ExcInternalError());

            dealii::SymmetricTensor<2,dim,real> hessian;
            // looping over each term of the homogenous polynomial
            for(unsigned int n = 0; n < n_vec; ++n){
                // comparing the indices values at each dimension pair
                // only thing assumed is indices[n] sums to 2 (order)
                for(unsigned int i = 0; i < dim; ++i){

                    // case 1: a*xi^2 (diagonal)
                    if((indices[n][i] == 2)){
                        hessian[i][i] = coeffs[n];
                    }

                    // case 2: a*xi*yi (off diagonal)
                    for(unsigned int j = i+1; j < dim; ++j){
                        if((indices[n][i] == 1) && (indices[n][j] == 1)){
                            hessian[i][j] = 0.5 * coeffs[n];
                        }
                    }

                }

            }

            // // debugging for dim = 2
            // for(unsigned int i = 0; i < n_vec; ++i)
            //     std::cout << "n_vec[" << i << "] = " << coeffs[i] << " * x^"<< indices[i][0] << " * y^" << indices[i][1] << std::endl;
            // std::cout << "Hessian = [" << hessian[0][0] << ", " << hessian[0][1] << "]" << std::endl;
            // std::cout << "          [" << hessian[1][0] << ", " << hessian[1][1] << "]" << std::endl << std::endl;

            // https://www.dealii.org/current/doxygen/deal.II/symmetric__tensor_8h.html#aa18a9d623fcd520f022421fd1d6c7a14
            using eigenpair = std::pair<real,dealii::Tensor<1,dim,real>>;
            std::array<eigenpair,dim> eig = dealii::eigenvectors(hessian); 
            
            // resorting the list based on the absolute value of the eigenvalue
            std::sort(eig.begin(), eig.end(), [](
                const eigenpair left,
                const eigenpair right)
            {
                return abs(left.first) > abs(right.first);
            });

            // storing the values
            for(int d = 0; d < dim; ++d){
                value_cell[d]     = abs(eig[d].first);
                direction_cell[d] = eig[d].second;
            }

        }else{

            // evaluating any point requires sum over power of the multindices
            auto eval = [&](const dealii::Tensor<1,dim,real>& point) -> real{
                real val = 0.0;
                for(unsigned int i = 0; i < n_vec; ++i){
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
                dealii::Tensor<1,dim,real> p_sample;
                for(unsigned int i = 0; i < n_sample; ++i){
                    theta = i*pi/n_sample;
                    
                    p_sample[0] = r*cos(theta);
                    p_sample[1] = r*sin(theta);
                    
                    val = abs(eval(p_sample));
                    if(val > A_max){
                        A_max = val;
                        t_max = theta;
                    }
                }

                dealii::Tensor<1,dim,real> p_1;
                p_1[0] = r*cos(t_max);
                p_1[1] = r*sin(t_max);

                // Taking A_2 to be at an angle of 90 degrees relative to first
                dealii::Tensor<1,dim,real> p_2;
                p_2[0] = r*cos(t_max+pi/2.0);
                p_2[1] = r*sin(t_max+pi/2.0);
                
                value_cell[0] = A_max;
                value_cell[1] = abs(eval(p_2));

                direction_cell[0] = p_1;
                direction_cell[1] = p_2;

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
                real y, r, phi, val;
                dealii::Tensor<1,dim,real> p_sample;
                for(unsigned int i = 0; i < n_sample_3d; ++i){
                    // calculation of the points 
                    y = (i*offset) - 1 + offset/2;
                    r = sqrt(1-pow(y,2));

                    phi = remainder(i, 2*n_sample_3d) * increment;

                    p_sample[0] = r*cos(phi);
                    p_sample[1] = y;
                    p_sample[2] = r*sin(phi);
                    
                    val = abs(eval(p_sample));
                    if(val > A_1){
                        A_1 = val;
                        p_1 = p_sample/p_sample.norm();
                    }
                }

                // generating the rest of the basis for p_1 rotation, two orthogonal vectors forming a plane
                dealii::Tensor<1,dim,real> u;
                dealii::Tensor<1,dim,real> v;

                // checking if near the x-axis
                dealii::Tensor<1,dim,real> px;
                px[0] = 1.0;
                if(abs(px*p_1) < 1.0/sqrt(2.0)){ // if further apart than 45 degrees use x-axis
                    // using cross products to generate two vectors orthogonal to p_1
                    u = dealii::cross_product_3d(p_1, px);
                }else{ // if not, use the y axis instead
                    dealii::Tensor<1,dim,real> py;
                    py[1] = 1.0;
                    u = dealii::cross_product_3d(p_1, py);
                }
                // second orthogonal to form the basis
                v = dealii::cross_product_3d(p_1, u);

                // normalizing 
                u = u / u.norm();
                v = v / v.norm();
                
                // now performing the 2d analysis in the plane uv
                real A_2 = 0.0, t_2 = 0.0;

                // using polar coordinates theta\in[0, \pi)
                real theta;
                dealii::Tensor<1,dim,real> p_2;
                for(unsigned int i = 0; i < n_sample; ++i){
                    theta = i*pi/n_sample;
                    p_2 = cos(theta)*u + sin(theta)*v;
                    
                    val = abs(eval(p_2));
                    if(val > A_2){
                        A_2 = val;
                        t_2 = theta;
                    }
                }

                // reassinging the largest value to p_2
                p_2 = cos(t_2)*u + sin(t_2)*v;

                // Taking A_2 to be at an angle of 90 degrees relative to first
                dealii::Tensor<1,dim,real> p_3 = cos(t_2+pi/2.0)*u + sin(t_2+pi/2.0)*v;
                
                // assigning the results
                value_cell[0] = A_1;
                value_cell[1] = A_2;
                value_cell[2] = abs(eval(p_3));

                direction_cell[0] = p_1;
                direction_cell[1] = p_2;
                direction_cell[2] = p_3;

            }else{ 
                
                // no other dimensions should appear
                Assert(false, dealii::ExcInternalError());

            }
        }

        // storing the tensor of results
        const unsigned int index = cell->active_cell_index(); 
        derivative_value[index]     = value_cell;
        derivative_direction[index] = direction_cell;
    }
}

template <int dim, int nstate, typename real>
void ReconstructPoly<dim,nstate,real>::reconstruct_manufactured_derivative(
    const std::shared_ptr<ManufacturedSolutionFunction<dim,real>>& manufactured_solution,
    const unsigned int                                             rel_order)
{
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        unsigned int order = cell->active_fe_index() + rel_order;
        assert(order == 2); // hessian based only
        (void) order;

        // evaluating the hessian from the manufactured solution
        dealii::Point<dim,real> center_point = cell->center();
        dealii::SymmetricTensor<2,dim,real> hessian = 
            manufactured_solution->hessian(center_point);

        // performing eigenvalue decomposition
        using eigenpair = std::pair<real,dealii::Tensor<1,dim,real>>;
        std::array<eigenpair,dim> eig = dealii::eigenvectors(hessian);

        std::sort(eig.begin(), eig.end(), [](
            const eigenpair left,
            const eigenpair right)
        {
            return abs(left.first) > abs(right.first);
        });

        // storing the values for the cell
        const unsigned int index = cell->active_cell_index(); 
        for(int d = 0; d < dim; ++d){
            derivative_value[index][d]     = abs(eig[d].first);
            derivative_direction[index][d] = eig[d].second;
        }

    }
}

// from DEALII, https://www.dealii.org/current/doxygen/deal.II/polynomial__space_8cc_source.html
// protected function so reimplementing slightly modified form here
// computes the multiindex for different dimensions, assuming the index map hasn't been modified
template <>
std::array<unsigned int, 1> compute_index<1>(
    const unsigned int i,
    const unsigned int /*size*/)
{
    return {{i}};
}

template <>
std::array<unsigned int, 2> compute_index<2>(
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
std::array<unsigned int, 3> compute_index<3>(
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

template <int dim, int nstate, typename real>
template <typename DoFCellAccessorType>
dealii::Vector<real> ReconstructPoly<dim,nstate,real>::reconstruct_norm(
    const NormType                                          norm_type,
    const DoFCellAccessorType &                             curr_cell,
    const dealii::PolynomialSpace<dim>                      ps,
    const dealii::LinearAlgebra::distributed::Vector<real> &solution)
{

    if(norm_type == NormType::H1){

        return reconstruct_H1_norm(
            curr_cell,
            ps,
            solution);

    }else if(norm_type == NormType::L2){

        return reconstruct_L2_norm(
            curr_cell,
            ps,
            solution);

    }else{

        // undefined
        assert(0);
        return dealii::Vector<real>(0);

    }

}

template <int dim, int nstate, typename real>
template <typename DoFCellAccessorType>
dealii::Vector<real> ReconstructPoly<dim,nstate,real>::reconstruct_H1_norm(
    const DoFCellAccessorType &                             curr_cell,
    const dealii::PolynomialSpace<dim>                      ps,
    const dealii::LinearAlgebra::distributed::Vector<real> &solution)
{

    // center point of the current cell
    dealii::Point<dim,real> center_point = curr_cell->center();

    // things to be extracted
    std::vector<std::array<real,nstate>>                       soln_at_q_vec;
    std::vector<std::array<dealii::Tensor<1,dim,real>,nstate>> grad_at_q_vec; // for the H^1 norm
    std::vector<dealii::Point<dim,real>>                       qpoint_vec;
    std::vector<real>                                          JxW_vec;

    // and keeping track of vector lengths
    unsigned int n_vec = 0;

    // fe_values
    dealii::hp::FEValues<dim,dim> fe_values_collection(
        mapping_collection,
        fe_collection,
        quadrature_collection,
        update_flags);

    // looping over the cell vector and extracting the soln, qpoint and JxW
    // std::vector<DoFCellAccessorType> cell_patch = dealii::GridTools::get_patch_around_cell(curr_cell);
    std::vector<DoFCellAccessorType> cell_patch = get_patch_around_dof_cell(curr_cell);
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
        std::array<real,nstate>                       soln_at_q;
        std::array<dealii::Tensor<1,dim,real>,nstate> grad_at_q; 
        for(unsigned int iquad = 0; iquad < n_quad; ++iquad){
            soln_at_q.fill(0.0);
            for(unsigned int istate = 0; istate < nstate; ++istate)
                grad_at_q[istate] = 0.0;
                        
            // looping over the DoFS to get the solution value
            for(unsigned int idof = 0; idof < n_dofs; ++idof){
                const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += solution[dofs_indices[idof]] * fe_values.shape_value_component(idof, iquad, istate);
                grad_at_q[istate] += solution[dofs_indices[idof]] * fe_values.shape_grad_component(idof, iquad, istate);
            }

            // moving the reference point to the center of the curr_cell
            dealii::Tensor<1,dim,real> tensor_q = fe_values.quadrature_point(iquad) - center_point;
            dealii::Point<dim,real> point_q(tensor_q);

            // push back into the vectors
            soln_at_q_vec.push_back(soln_at_q);
            grad_at_q_vec.push_back(grad_at_q);
            qpoint_vec.push_back(point_q);
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
    mat = 0.0;
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

        // take inner product of \psi_i and u (solution)
        // if multiple states, taking the 2 norm of the different states
        dealii::Vector<real> rhs_poly(nstate);
        for(unsigned int istate = 0; istate < nstate; ++istate)
            for(unsigned int i_vec = 0; i_vec < n_vec; ++i_vec)
                rhs[i_poly] += ((ps.compute_value(i_poly, qpoint_vec[i_vec]) * soln_at_q_vec[i_vec][istate])
                               +(ps.compute_grad(i_poly, qpoint_vec[i_vec]) * grad_at_q_vec[i_vec][istate]))
                               *JxW_vec[i_vec];
    }

    // solving the system
    dealii::Vector<real> coeffs(n_poly);

    mat.gauss_jordan();
    mat.vmult(coeffs, rhs);

    return coeffs;
}

template <int dim, int nstate, typename real>
template <typename DoFCellAccessorType>
dealii::Vector<real> ReconstructPoly<dim,nstate,real>::reconstruct_L2_norm(
    const DoFCellAccessorType &                             curr_cell,
    const dealii::PolynomialSpace<dim>                      ps,
    const dealii::LinearAlgebra::distributed::Vector<real> &solution)
{
    // center point of the current cell
    dealii::Point<dim,real> center_point = curr_cell->center();

    // things to be extracted
    std::vector<std::array<real,nstate>>                       soln_at_q_vec;
    std::vector<dealii::Point<dim,real>>                       qpoint_vec;
    std::vector<real>                                          JxW_vec;

    // and keeping track of vector lengths
    unsigned int n_vec = 0;

    // fe_values
    dealii::hp::FEValues<dim,dim> fe_values_collection(
        mapping_collection,
        fe_collection,
        quadrature_collection,
        update_flags);

    // looping over the cell vector and extracting the soln, qpoint and JxW
    // std::vector<DoFCellAccessorType> cell_patch = dealii::GridTools::get_patch_around_cell(curr_cell);
    std::vector<DoFCellAccessorType> cell_patch = get_patch_around_dof_cell(curr_cell);
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
        std::array<real,nstate> soln_at_q;
        for(unsigned int iquad = 0; iquad < n_quad; ++iquad){
            soln_at_q.fill(0.0);
                        
            // looping over the DoFS to get the solution value
            for(unsigned int idof = 0; idof < n_dofs; ++idof){
                const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += solution[dofs_indices[idof]] * fe_values.shape_value_component(idof, iquad, istate);
            }

            // moving the reference point to the center of the curr_cell
            dealii::Tensor<1,dim,real> tensor_q = fe_values.quadrature_point(iquad) - center_point;
            dealii::Point<dim,real> point_q(tensor_q);

            // push back into the vectors
            soln_at_q_vec.push_back(soln_at_q);
            qpoint_vec.push_back(point_q);
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
    mat = 0.0;
    for(unsigned int i_poly = 0; i_poly < n_poly; ++i_poly){
        for(unsigned int j_poly = 0; j_poly < n_poly; ++j_poly){
            // taking the inner product between \psi_i and \psi_j
            // <u,v>_{L^2(\Omega)} = \int_{\Omega} u*v dx
            for(unsigned int i_vec = 0; i_vec < n_vec; ++i_vec)
                mat.add(i_poly, j_poly, 
                    (ps.compute_value(i_poly, qpoint_vec[i_vec]) * ps.compute_value(j_poly, qpoint_vec[i_vec]))
                    *JxW_vec[i_vec]);
        }

        // take inner product of \psi_i and u (solution)
        // if multiple states, taking the 2 norm of the different states
        dealii::Vector<real> rhs_poly(nstate);
        for(unsigned int istate = 0; istate < nstate; ++istate)
            for(unsigned int i_vec = 0; i_vec < n_vec; ++i_vec)
                rhs[i_poly] += (ps.compute_value(i_poly, qpoint_vec[i_vec]) * soln_at_q_vec[i_vec][istate])
                               *JxW_vec[i_vec];
    }

    // solving the system
    dealii::Vector<real> coeffs(n_poly);

    mat.gauss_jordan();
    mat.vmult(coeffs, rhs);

    return coeffs;
}

// based on DEALII GridTools::get_patch_around_cell
// https://www.dealii.org/current/doxygen/deal.II/grid__tools__dof__handlers_8cc_source.html#l01411
// modified to work directly on the dof_handler accesor for hp-access rather than casting back and forth
template <int dim, int nstate, typename real>
template <typename DoFCellAccessorType>
std::vector<DoFCellAccessorType> ReconstructPoly<dim,nstate,real>::get_patch_around_dof_cell(
    const DoFCellAccessorType &cell)
{
    Assert(cell->is_locally_owned(), dealii::ExcInternalError());

    std::vector<DoFCellAccessorType> patch;
    patch.push_back(cell);

    for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
        if(cell->face(iface)->at_boundary()) continue;

        Assert(cell->neighbor(iface).state() == dealii::IteratorState::valid,
                   dealii::ExcInternalError());

        if(cell->neighbor(iface)->has_children() == false){ // case 1: coarse cell
            patch.push_back(cell->neighbor(iface));
        }else{ // has children cells
            if(dim > 1){ // case 2: (2d/3d) get subface cells
                for(unsigned int subface = 0; subface < cell->face(iface)->n_children(); ++subface)
                    patch.push_back(cell->neighbor_child_on_subface(iface, subface));
            }else{ // case 3: (1d) iterate over children to find one on boundary
                DoFCellAccessorType neighbor = cell->neighbor(iface);

                // looping over the children
                while(neighbor->has_children())
                    neighbor = neighbor->child(1-iface);

                Assert(neighbor->neighbor(1-iface) == cell, dealii::ExcInternalError());
                patch.push_back(neighbor);
            }
        }
    }

    return patch;
}

template <int dim, int nstate, typename real>
dealii::Vector<real> ReconstructPoly<dim,nstate,real>::get_derivative_value_vector_dealii(
    const unsigned int index)
{
    dealii::Vector<real> vec(derivative_value.size());

    for(unsigned int i = 0; i < derivative_value.size(); i++){
        vec[i] = derivative_value[i][index];
    }

    return vec;
}

template class ReconstructPoly<PHILIP_DIM, 1, double>;
template class ReconstructPoly<PHILIP_DIM, 2, double>;
template class ReconstructPoly<PHILIP_DIM, 3, double>;
template class ReconstructPoly<PHILIP_DIM, 4, double>;
template class ReconstructPoly<PHILIP_DIM, 5, double>;

} // namespace GridRefinement

} // namespace PHiLiP
