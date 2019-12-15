#include <iostream>
#include <algorithm>
#include <functional>

#include <Sacado.hpp>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include "physics/manufactured_solution.h"

#include "size_field.h"

namespace PHiLiP {

namespace GridRefinement {
// functions for computing the target size field over a domain,
// need various implementations for Hessian-exact, reconstructed quadratic
// adjoint based and then hp-cases

// eventually add a parameter file to select how to construct the size field
// takes as input a target complexity, computes the h(x,y) (target isotropic 
// size field) and outputs to a cell-wise vector for gmsh_out
template <int dim, typename real>
void SizeField<dim,real>::isotropic_uniform(
    const dealii::Triangulation<dim, dim> &tria,                             // triangulation
    const dealii::Mapping<dim, dim > &     mapping,                          // mapping field used in computed JxW
    const dealii::FiniteElement<dim, dim> &fe,                               // for fe_values integration, assumed constant for now
    std::shared_ptr< PHiLiP::ManufacturedSolutionFunction<dim,real> > 
                                           manufactured_solution_function,   // manufactured solution
    real                                   complexity,                       // continuous dof measure
    dealii::Vector<real> &                 h_field)                          // output vector is the 1D sizes
{
    typename dealii::Triangulation<dim, dim>::active_cell_iterator cell =
        tria.begin_active();
    const typename dealii::Triangulation<dim, dim>::active_cell_iterator endc =
        tria.end();

    // for a uniform polynomial distribution
    real q = 2; // L2 norm for the error
    real p = 1; // for now assuming linear elements

    // calculating the scaling parameter on each cell
    // B = (A_1 A_2)^{q/2} where A_1 and A_2 are the eigenvalues of the quadratic error model
    dealii::Vector<real> B(tria.n_active_cells()); 
    for(cell = tria.begin_active(); cell!=endc; ++cell){
        if(!cell->is_locally_owned()) continue;

        // getting the central coordinate as average of vertices
        dealii::Point<dim> pos;
        unsigned int vertices_per_cell = dealii::GeometryInfo<dim>::vertices_per_cell;
        for(unsigned int vertex = 0; vertex < vertices_per_cell; ++vertex){
            // adding the contributions from each of the nodes
            pos += cell->vertex(vertex);
        }
        // averaging
        pos /= vertices_per_cell;

        // evaluating the Hessian at this point
        dealii::SymmetricTensor<2,dim,real> H = 
            manufactured_solution_function->hessian(pos); // using default state

        std::cout << "H=[" << H[0][0] << ", " << H[1][0] << '\n'
                  << "   " << H[0][1] << ", " << H[1][1] << "]" << '\n';

        // assuming in 2D for now
        if(dim == 2){
            // // A_1, A_2 are abs of eigenvalues
            // // TODO: for p=1 only
            // real A1 = 0.5*abs((H[0][0] + H[1][1]) + sqrt(pow(H[0][0] + H[1][1], 2) - 4.0*(H[0][0]*H[1][1] - H[0][1]*H[1][0])));
            // real A2 = 0.5*abs((H[0][0] + H[1][1]) - sqrt(pow(H[0][0] + H[1][1], 2) - 4.0*(H[0][0]*H[1][1] - H[0][1]*H[1][0])));
            // std::cout << "A1 = " << A1 << '\n'
            //           << "A2 = " << A2 << '\n';
            // B[cell->active_cell_index()] = pow(A1*A2, q/2);

            // product of eigenvalues should just be the detemrinant
            B[cell->active_cell_index()] = pow(abs(H[0][0]*H[1][1] - H[0][1]*H[1][0]), q/2);
            std::cout << "B[" << cell->active_cell_index() << "]=" << B[cell->active_cell_index()] << std::endl;
        }
    }
    // taking the integral over the domain (piecewise constant)
    dealii::QGauss<dim> quadrature(1);
    dealii::FEValues<dim,dim> fe_values(mapping, fe, quadrature, dealii::update_JxW_values);
    
    // leaving this in to be general for now
    const unsigned int n_quad_pts = fe_values.n_quadrature_points;

    // complexity per cell (based on polynomial orer)
    real w = pow(p+1, dim);

    real exponent = 2.0/((p+1)*q+2.0);

    // integral value
    real integral_value = 0;
    for(cell = tria.begin_active(); cell!=endc; ++cell){
        if(!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);

        // value of the B const
        for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad)
            integral_value += w * pow(B[cell->active_cell_index()], exponent) * fe_values.JxW(iquad);
    }

    // constant now known (since q and p are uniform, otherwise would be a function of p and w)
    real K = complexity/integral_value;

    // looping over the elements to define the sizes
    h_field.reinit(tria.n_active_cells());
    for(cell = tria.begin_active(); cell!=endc; ++cell){
        if(!cell->is_locally_owned()) continue;

        h_field[cell->active_cell_index()] = pow(K*pow(B[cell->active_cell_index()], exponent), -1.0/dim);
    }
}

template <int dim, typename real>
void SizeField<dim,real>::isotropic_h(
    real                                 complexity,            // (input) complexity target
    dealii::Vector<real> &               B,                     // only one since p is constant
    dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
    dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    dealii::Vector<real> &               h_field,               // (output) size field
    dealii::Vector<real> &               p_field)               // (input)  poly field
{
    // setting up lambda function which, given a constant for the size field, 
    // updates the h distribution and outputs the new complexity
    auto f = [&](real lam) -> real{
        update_h_optimal(lam, B, dof_handler, h_field, p_field);
        real current_complexity = evaluate_complexity(dof_handler, mapping_collection, fe_collection, quadrature_collection, update_flags, h_field, p_field);
        return current_complexity - complexity;
    };

    // call to the optimization (bisection)
    real a = 0;
    real b = 1000;
    real lam = bisection(f, a, b);

    // final update with converged parameter
    update_h_optimal(lam, B, dof_handler, h_field, p_field);
}

template <int dim, typename real>
real SizeField<dim,real>::evaluate_complexity(
    dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
    dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    dealii::Vector<real> &               h_field,               // (output) size field
    dealii::Vector<real> &               p_field)               // (input)  poly field    
{
    real complexity_sum = 0.0;

    // fe_values
    dealii::hp::FEValues<dim,dim> fe_values_collection(
        mapping_collection,
        fe_collection,
        quadrature_collection,
        update_flags);

    // evaluate the complexity of a provided field
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        const unsigned int index = cell->active_cell_index();

        const unsigned int mapping_index = 0;
        const unsigned int fe_index = cell->active_fe_index();
        const unsigned int quad_index = fe_index;

        const unsigned int n_quad = quadrature_collection[quad_index].size();

        fe_values_collection.reinit(cell, quad_index, mapping_index, fe_index); 
        const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();

        real JxW = 0;
        for(unsigned int iquad = 0; iquad < n_quad; ++iquad)
            JxW += fe_values.JxW(iquad);

        complexity_sum += pow((p_field[index]+1)/h_field[index], dim) * JxW;
    }

    return dealii::Utilities::MPI::sum(complexity_sum, MPI_COMM_WORLD);
}

template <int dim, typename real>
void SizeField<dim,real>::update_h_optimal(
    real                          lam,         // (input) bisection parameter
    dealii::Vector<real> &        B,           // constant for current p
    dealii::hp::DoFHandler<dim> & dof_handler, // dof_handler
    dealii::Vector<real> &        h_field,     // (output) size field
    dealii::Vector<real> &        p_field)     // (input)  poly field
{
    real q = 2.0;

    // looping over the cells and updating the values
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        const unsigned int index = cell->active_cell_index();

        const real p = p_field[index];

        const real exponent  = -1.0/(q*(p+1)+2.0);
        const real component = q*(p+1.0)/(q*(p+1)+2.0) * B[index]/pow(p+1, dim);

        h_field[index] = lam * pow(exponent, component);
    }
}

template <int dim, typename real>
void SizeField<dim,real>::isotropic_p(
    dealii::Vector<real> &               Bm,                    // constant for p-1
    dealii::Vector<real> &               B,                     // constant for p
    dealii::Vector<real> &               Bp,                    // constant for p+1
    dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
    dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    dealii::Vector<real> &               h_field,               // (input) size field
    dealii::Vector<real> &               p_field)               // (output) poly field
{
    (void)Bm;
    (void)B;
    (void)Bp;
    (void)dof_handler;
    (void)mapping_collection;
    (void)fe_collection;
    (void)quadrature_collection;
    (void)update_flags;
    (void)h_field;
    (void)p_field;

    // this is like its own thing, might have to do the integration or something
    // because we can't adjust the local h to compensate for p increases, needs 
    // to be adjusted as a global system

    // maybe: start from current, check dofs/error and adjust to anything above/below the average
    // refine progressively if complexity is below the limit
    // coarsen progressively if above
    // need some way of measuring when stability is reached. Maybe can be done as a bulk criterion
}

template <int dim, typename real>
void SizeField<dim,real>::isotropic_hp(
    real                                 complexity,            // complexity target
    dealii::Vector<real> &               Bm,                    // constant for p-1
    dealii::Vector<real> &               B,                     // constant for p
    dealii::Vector<real> &               Bp,                    // constant for p+1
    dealii::hp::DoFHandler<dim> &        dof_handler,           // dof_handler
    dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    dealii::Vector<real> &               h_field,               // (output) size field
    dealii::Vector<real> &               p_field)               // (output) poly field
{
    isotropic_h(
        complexity,
        B,
        dof_handler,
        mapping_collection,
        fe_collection,
        quadrature_collection,
        update_flags,
        h_field,
        p_field);

    real q = 1.0;

    // two options here, either constant error (preferable) 
    // or constant complexity target (Dolejsi)
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        unsigned int index = cell->active_cell_index();

        // computing the reference error
        real e_ref = pow(abs(B[index]), q) 
                   * pow(h_field[index], dim*q*(p_field[index]+1)/2);
        
        // local complexity
        real N_ref = pow((p_field[index]+1)/h_field[index], dim);

        // constant error
        // computing the error for increased/decreased p with the same local complexity
        real h_m = (p_field[index])  /pow(N_ref, 1.0/dim);
        real h_p = (p_field[index]+2)/pow(N_ref, 1.0/dim);

        // computing the local error for each
        real e_m = pow(abs(Bm[index]), q) 
                 * pow(h_m, dim*q*(p_field[index]  )/2);
        real e_p = pow(abs(Bp[index]), q) 
                 * pow(h_p, dim*q*(p_field[index]+2)/2);
        
        // determining which is the smallest and adjusting
        if(e_m < e_ref && e_m <= e_p){
            h_field[index] = h_m;
            p_field[index]--;
        }else if(e_p < e_ref && e_p <= e_m){
            h_field[index] = h_p;
            p_field[index]++;
        }// else, p stays the same
    }

    // either add a check here for which is obtained or add an output for the +/-
    // only if another change needs to be made here
}

// functions for solving non-linear problems
template <int dim, typename real>
real SizeField<dim,real>::bisection(
    std::function<real(real)> func,  
    real                      lower_bound, 
    real                      upper_bound)
{
    real f_lb = func(lower_bound);
    real f_ub = func(upper_bound);

    // f_ub is unused before being reset if not present here
    AssertThrow(f_lb * f_ub < 0, dealii::ExcInternalError());

    real x   = (lower_bound + upper_bound)/2.0;
    real f_x = func(x);

    real         tolerance = 1e-6;
    unsigned int max_iter  = 1000;

    unsigned int i = 0;
    while(f_x > tolerance && i < max_iter){
        if(f_x * f_lb < 0){
            upper_bound = x;
            f_ub        = f_x;
        }else{
            lower_bound = x;
            f_lb        = f_x;
        }

        x   = (lower_bound + upper_bound)/2.0;
        f_x = func(x);

        i++;
    }

    Assert(i < max_iter, dealii::ExcInternalError());

    return x;
}


template class SizeField <PHILIP_DIM, double>;
// template class SizeField <PHILIP_DIM, float>; // manufactured solution isn't defined for this

} // namespace GridRefinement

} // namespace PHiLiP
