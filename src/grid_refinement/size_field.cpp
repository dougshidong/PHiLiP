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

#include "grid_refinement/field.h"
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
    const real &                               complexity,  // (input) complexity target
    const dealii::Vector<real> &               B,           // only one since p is constant
    const dealii::DoFHandler<dim> &            dof_handler, // dof_handler
    std::unique_ptr<Field<dim,real>> &         h_field,     // (output) size field
    const real &                               poly_degree) // (input)  polynomial degree
{
    const real q = 2.0; 
    const real exponent = 2.0/((poly_degree+1)*q+2.0);

    // integral value
    real integral_value = 0.0;
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            integral_value += pow(B[cell->active_cell_index()], exponent) * cell->measure();

    // complexity per cell (based on polynomial orer)
    integral_value *= pow(poly_degree+1, dim);

    real integral_value_mpi = dealii::Utilities::MPI::sum(integral_value, MPI_COMM_WORLD);

    // constant known (since q and p are uniform, otherwise would be a function of p and w)
    const real K = complexity/integral_value_mpi;

    // looping over the elements to define the sizes
    h_field->reinit(dof_handler.get_triangulation().n_active_cells());
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            h_field->set_scale(cell->active_cell_index(), pow(K*pow(B[cell->active_cell_index()], exponent), -1.0/dim));
}

template <int dim, typename real>
void SizeField<dim,real>::isotropic_h(
    const real                                 complexity,            // (input) complexity target
    const dealii::Vector<real> &               B,                     // only one since p is constant
    const dealii::DoFHandler<dim> &            dof_handler,           // dof_handler
    const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    std::unique_ptr<Field<dim,real>> &         h_field,               // (output) size field
    const dealii::Vector<real> &               p_field)               // (input)  poly field
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
    const dealii::DoFHandler<dim> &            dof_handler,           // dof_handler
    const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    const std::unique_ptr<Field<dim,real>> &   h_field,               // (input) size field
    const dealii::Vector<real> &               p_field)               // (input)  poly field    
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

        complexity_sum += pow((p_field[index]+1)/h_field->get_scale(index), dim) * JxW;
    }

    return dealii::Utilities::MPI::sum(complexity_sum, MPI_COMM_WORLD);
}

template <int dim, typename real>
void SizeField<dim,real>::update_h_optimal(
    const real                          lam,         // (input) bisection parameter
    const dealii::Vector<real> &        B,           // constant for current p
    const dealii::DoFHandler<dim> &     dof_handler, // dof_handler
    std::unique_ptr<Field<dim,real>> &  h_field,     // (output) size field
    const dealii::Vector<real> &        p_field)     // (input)  poly field
{
    const real q = 2.0;

    // looping over the cells and updating the values
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        const unsigned int index = cell->active_cell_index();

        const real p = p_field[index];

        const real exponent  = -1.0/(q*(p+1)+2.0);
        const real component = q*(p+1.0)/(q*(p+1)+2.0) * B[index]/pow(p+1, dim);

        h_field->set_scale(index, lam * pow(exponent, component));
    }
}

// computes updated p-field with a constant h-field
// NOT IMPLEMENTED yet
/*
template <int dim, typename real>
void SizeField<dim,real>::isotropic_p(
    const dealii::Vector<real> &               Bm,                    // constant for p-1
    const dealii::Vector<real> &               B,                     // constant for p
    const dealii::Vector<real> &               Bp,                    // constant for p+1
    const dealii::DoFHandler<dim> &            dof_handler,           // dof_handler
    const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    const std::unique_ptr<Field<dim,real>> &   h_field,               // (input) size field
    dealii::Vector<real> &                     p_field)               // (output) poly field
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
*/

template <int dim, typename real>
void SizeField<dim,real>::isotropic_hp(
    const real                                 complexity,            // complexity target
    const dealii::Vector<real> &               Bm,                    // constant for p-1
    const dealii::Vector<real> &               B,                     // constant for p
    const dealii::Vector<real> &               Bp,                    // constant for p+1
    const dealii::DoFHandler<dim> &            dof_handler,           // dof_handler
    const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    std::unique_ptr<Field<dim,real>> &         h_field,               // (output) size field
    dealii::Vector<real> &                     p_field)               // (output) poly field
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

    const real q = 2.0;

    // two options here, either constant error (preferable) 
    // or constant complexity target (Dolejsi)
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        unsigned int index = cell->active_cell_index();

        // computing the reference error
        const real e_ref = pow(abs(B[index]), q) 
                         * pow(h_field->get_scale(index), dim*q*(p_field[index]+1)/2);
        
        // local complexity
        const real N_ref = pow((p_field[index]+1)/h_field->get_scale(index), dim);

        // constant error
        // computing the error for increased/decreased p with the same local complexity
        const real h_m = (p_field[index])  /pow(N_ref, 1.0/dim);
        const real h_p = (p_field[index]+2)/pow(N_ref, 1.0/dim);

        // computing the local error for each
        const real e_m = pow(abs(Bm[index]), q) 
                       * pow(h_m, dim*q*(p_field[index]  )/2);
        const real e_p = pow(abs(Bp[index]), q) 
                       * pow(h_p, dim*q*(p_field[index]+2)/2);
        
        // determining which is the smallest and adjusting
        if(e_m < e_ref && e_m <= e_p){
            h_field->set_scale(index, h_m);
            p_field[index]--;
        }else if(e_p < e_ref && e_p <= e_m){
            h_field->set_scale(index, h_p);
            p_field[index]++;
        }// else, p stays the same
    }

    // either add a check here for which is obtained or add an output for the +/-
    // only if another change needs to be made here
}

template <int dim, typename real>
void SizeField<dim,real>::adjoint_uniform_balan(
    const real                                 complexity,            // target complexity
    const real                                 r_max,                 // maximum refinement factor
    const real                                 c_max,                 // maximum coarsening factor
    const dealii::Vector<real> &               eta,                   // error indicator (DWR)
    const dealii::DoFHandler<dim> &            dof_handler,           // dof_handler
    const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    std::unique_ptr<Field<dim,real>>&          h_field,               // (output) target size_field
    const real &                               poly_degree)           // uniform polynomial degree
{
    // creating a proper polynomial vector
    dealii::Vector<real> p_field(dof_handler.get_triangulation().n_active_cells());
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            p_field[cell->active_cell_index()] = poly_degree;

    // calling the regular version of adjoint_h
    adjoint_h_balan(
        complexity,
        r_max,
        c_max,
        eta,
        dof_handler,
        mapping_collection,
        fe_collection,
        quadrature_collection,
        update_flags,
        h_field,
        p_field);
}

template <int dim, typename real>
void SizeField<dim,real>::adjoint_h_balan(
    const real                                 complexity,            // target complexity
    const real                                 r_max,                 // maximum refinement factor
    const real                                 c_max,                 // maximum coarsening factor
    const dealii::Vector<real> &               eta,                   // error indicator (DWR)
    const dealii::DoFHandler<dim> &            dof_handler,           // dof_handler
    const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    std::unique_ptr<Field<dim,real>>&          h_field,               // (output) target size_field
    const dealii::Vector<real> &               p_field)               // polynomial degree vector
{
    // getting the I_c vector of the initial sizes
    dealii::Vector<real> I_c(dof_handler.get_triangulation().n_active_cells());
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            I_c[cell->active_cell_index()] = pow(h_field->get_scale(cell->active_cell_index()), (real)dim);

    // getting minimum and maximum of eta
    real eta_min_local, eta_max_local;
    bool first = true;
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        unsigned int index = cell->active_cell_index();

        // for first cell found, setting the value 
        if(first){
            eta_min_local = eta[index];
            eta_max_local = eta[index];
            first = false;
        }else{

            // performing checks
            if(eta[index] < eta_min_local)
                eta_min_local = eta[index];

            if(eta[index] > eta_max_local)
                eta_max_local = eta[index];

        }
    }

    // each processor needs atleast 1 cell
    Assert(first, dealii::ExcInternalError());

    // perform mpi call
    real eta_min = dealii::Utilities::MPI::min(eta_min_local, MPI_COMM_WORLD);
    real eta_max = dealii::Utilities::MPI::max(eta_max_local, MPI_COMM_WORLD);

    real initial_complexity = evaluate_complexity(
            dof_handler, 
            mapping_collection, 
            fe_collection, 
            quadrature_collection, 
            update_flags, 
            h_field, 
            p_field);
    std::cout << "Starting complexity = " << initial_complexity << std::endl;
    std::cout << "Target complexity = " << complexity << std::endl;
    std::cout << "f_0 = " << (initial_complexity - complexity) << std::endl;

    // setting up the bisection functional, based on an input value of
    // eta_ref, determines the complexity value for the mesh (using DWR estimates
    // weighted in the quadratic logarithmic space).
    auto f = [&](real eta_ref) -> real{
        // updating the size field
        update_alpha_vector_balan(
            eta,
            r_max, 
            c_max, 
            eta_min, 
            eta_max,
            eta_ref,
            dof_handler,
            I_c,
            h_field);

        // getting the complexity and returning the difference with the target
        real current_complexity = evaluate_complexity(
            dof_handler, 
            mapping_collection, 
            fe_collection, 
            quadrature_collection, 
            update_flags, 
            h_field, 
            p_field);
        return current_complexity - complexity;
    };

    // call to optimization (bisection), using min and max as initial bounds
    real eta_target = bisection(f, eta_max, eta_min);
    std::cout << "Bisection finished with eta_ref = "<< eta_target << ", f(eta_ref)=" << f(eta_target) << std::endl;

    // final uppdate using the converged parameter
    update_alpha_vector_balan(
        eta,
        r_max, 
        c_max, 
        eta_min, 
        eta_max,
        eta_target,
        dof_handler,
        I_c,
        h_field);

}

// performs adjoint based size field adaptatation with uniform p-field
// peforms equidistribution of DWR to sizes based on 2p+1 power of convergence
template <int dim, typename real>
void SizeField<dim,real>::adjoint_h_equal(
    const real                                 complexity,            // target complexity
    const dealii::Vector<real> &               eta,                   // error indicator (DWR)
    const dealii::DoFHandler<dim> &            dof_handler,           // dof_handler
    const dealii::hp::MappingCollection<dim> & mapping_collection,    // mapping collection
    const dealii::hp::FECollection<dim> &      fe_collection,         // fe collection
    const dealii::hp::QCollection<dim> &       quadrature_collection, // quadrature collection
    const dealii::UpdateFlags &                update_flags,          // update flags for for volume fe
    std::unique_ptr<Field<dim,real>>&          h_field,               // (output) target size_field
    const real &                               poly_degree)           // uniform polynomial degree
{
    std::cout << "Starting equal distribution of DWR based on 2p+1 power." << std::endl;

    // creating a proper polynomial vector
    dealii::Vector<real> p_field(dof_handler.get_triangulation().n_active_cells());
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            p_field[cell->active_cell_index()] = poly_degree;

    // getting minimum and maximum of eta
    real eta_min_local, eta_max_local;
    bool first = true;
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        unsigned int index = cell->active_cell_index();

        // for first cell found, setting the value 
        if(first){
            eta_min_local = eta[index];
            eta_max_local = eta[index];
            first = false;
        }else{

            // performing checks
            if(eta[index] < eta_min_local)
                eta_min_local = eta[index];

            if(eta[index] > eta_max_local)
                eta_max_local = eta[index];

        }
    }

    // each processor needs atleast 1 cell
    Assert(first, dealii::ExcInternalError());

    // perform mpi call
    real eta_min = dealii::Utilities::MPI::min(eta_min_local, MPI_COMM_WORLD);
    real eta_max = dealii::Utilities::MPI::max(eta_max_local, MPI_COMM_WORLD);

    // setting up the bisection functional, based on an input value of tau
    // determines new size from 2p+1 root relative to local DWR
    auto f = [&](real tau) -> real{
        // updating the size field
        update_h_dwr(
            tau,
            eta, 
            dof_handler,
            h_field,
            poly_degree);

        // getting the complexity and returning the difference with the target
        real current_complexity = evaluate_complexity(
            dof_handler, 
            mapping_collection, 
            fe_collection, 
            quadrature_collection, 
            update_flags, 
            h_field, 
            p_field);
        return current_complexity - complexity;
    };

    // performing the bisection call
    real tau_target = bisection(f, eta_min, eta_max, 1e-10);

    // updating the size field
    update_h_dwr(
        tau_target,
        eta, 
        dof_handler,
        h_field,
        poly_degree);
}

// sets the h_field sizes based on a reference value and DWR distribution
template <int dim, typename real>
void SizeField<dim,real>::update_h_dwr(
    const real                          tau,         // reference value for settings sizes
    const dealii::Vector<real> &        eta,         // error indicator (DWR)
    const dealii::DoFHandler<dim> &     dof_handler, // dof_handler
    std::unique_ptr<Field<dim,real>>&   h_field,     // (output) target size_field
    const real &                        poly_degree) // uniform polynomial degree
{
    // exponent for inverse DWR scaling
    const real Nrt = 1.0/(2*poly_degree+1);

    // looping through the cells and updating their size using tau
     for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        unsigned int index = cell->active_cell_index();

        // getting the new length based on Nrt
        const real h_target = pow(tau/eta[index], Nrt);

        // applying in the field
        h_field->set_scale(index, h_target);
    }

}

// updates the size targets for the entire mesh (from alpha)
// based on the input of a bisection parameter eta_ref
template <int dim, typename real>
void SizeField<dim,real>::update_alpha_vector_balan(
    const dealii::Vector<real>&        eta,         // vector of DWR indicators
    const real                         r_max,       // max refinement factor
    const real                         c_max,       // max coarsening factor
    const real                         eta_min,     // minimum DWR
    const real                         eta_max,     // maximum DWR
    const real                         eta_ref,     // reference parameter for bisection
    const dealii::DoFHandler<dim>&     dof_handler, // dof_handler
    const dealii::Vector<real>&        I_c,         // cell area measure
    std::unique_ptr<Field<dim,real>>&  h_field)     // (output) size-field
{
    // looping through the cells and updating their size (h_field)
    for(auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        unsigned int index = cell->active_cell_index();

        // getting the alpha factor for the cell update
        real alpha_k = update_alpha_k_balan(
            eta[index],
            r_max,
            c_max,
            eta_min,
            eta_max,
            eta_ref);

        // getting the new length based on I_c (cell area)
        real h_target = pow(alpha_k * I_c[index], 1.0/dim);

        // real h = h_field->get_scale(index);
        // std::cout << "h = " << h << ", h_t = " << h_target << ", alpha = " << alpha_k << std::endl;
        // std::cout << "I_c = " << I_c[index] << ", sqrt(I_c) = " << pow(I_c[index], 1.0/dim) << std::endl;;

        // updating the h_field
        h_field->set_scale(index, h_target);

    }
}

// function that determines local alpha size refinement factor (from adjoint estimates)
// from eq. 30-33 of Balan et al. "djoint-based hp-adaptivity on anisotropic meshes for high-order..."
template <int dim, typename real>
real SizeField<dim,real>::update_alpha_k_balan(
    const real eta_k,   // local DWR factor
    const real r_max,   // maximum refinement factor
    const real c_max,   // maximum coarsening factor
    const real eta_min, // minimum DWR indicator
    const real eta_max, // maximum DWR indicator
    const real eta_ref) // referebce DWR for determining coarsening/refinement
{
    // considering two possible cases (above or below reference)
    // also need to check close to equality to avoid divide by ~= 0
    real alpha_k;
    
    if(eta_k > eta_ref){ 

        // getting the quadratic coefficient
        real xi_k = (log(eta_k) - log(eta_ref)) / (log(eta_max) - log(eta_ref));


        // getting the refinement factor
        alpha_k = 1.0 / ((r_max-1)*xi_k*xi_k + 1.0);

    }else if(eta_k < eta_ref){

        // getting the quadratic coefficient
        real xi_k = (log(eta_k) - log(eta_ref)) / (log(eta_min) - log(eta_ref));

        // getting the coarsening factor
        alpha_k = ((c_max-1)*xi_k*xi_k + 1.0);

    }else{

        // performing no change (right on)
        alpha_k = 1.0;

    }

    // update area fraction (relative to initial area)
    return alpha_k;
}

// functions for solving non-linear problems
template <int dim, typename real>
real SizeField<dim,real>::bisection(
    const std::function<real(real)> func,  
    real                            lower_bound, 
    real                            upper_bound,
    real                            rel_tolerance,
    real                            abs_tolerance)
{
    real f_lb = func(lower_bound);
    real f_ub = func(upper_bound);

    std::cout << "lb = " << lower_bound << ", f_lb = " << f_lb << std::endl;
    std::cout << "ub = " << upper_bound << ", f_ub = " << f_ub << std::endl;

    // f_ub is unused before being reset if not present here
    AssertThrow(f_lb * f_ub < 0, dealii::ExcInternalError());

    real x   = (lower_bound + upper_bound)/2.0;
    real f_x = func(x);

    // const real         rel_tolerance = 1e-6;
    const unsigned int max_iter = 1000;

    real tolerance = rel_tolerance * abs(f_ub-f_lb);
    if(abs_tolerance < tolerance)
        tolerance = abs_tolerance;

    unsigned int i = 0;
    while(abs(f_x) > tolerance && i < max_iter){
        if(f_x * f_lb < 0){
            upper_bound = x;
            f_ub        = f_x;
        }else{
            lower_bound = x;
            f_lb        = f_x;
        }

        x   = (lower_bound + upper_bound)/2.0;
        f_x = func(x);

        std::cout << "iter #" << i << ", x = " << x << ", fx = " << f_x << std::endl;

        i++;
    }

    Assert(i < max_iter, dealii::ExcInternalError());

    return x;
}

template class SizeField <PHILIP_DIM, double>;
// template class SizeField <PHILIP_DIM, float>; // manufactured solution isn't defined for this

} // namespace GridRefinement

} // namespace PHiLiP
