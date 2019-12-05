#include <iostream>

#include <Sacado.hpp>

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

template class SizeField <PHILIP_DIM, double>;
// template class SizeField <PHILIP_DIM, float>; // manufactured solution isn't defined for this

} // namespace GridRefinement

} // namespace PHiLiP
