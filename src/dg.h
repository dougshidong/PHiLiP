#ifndef __DISCONTINUOUSGALERKIN_H__
#define __DISCONTINUOUSGALERKIN_H__

#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/mapping_q1.h> // Might need mapping_q
#include <deal.II/fe/mapping_q.h> // Might need mapping_q

#include <deal.II/dofs/dof_handler.h>


#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <Sacado.hpp>

#include "physics/physics.h"
#include "numerical_flux.h"
#include "parameters.h"


namespace PHiLiP
{
    using namespace dealii;

    template <int dim, int nstate, typename real>
    class DiscontinuousGalerkin // : public DiscontinuousGalerkin
    {
    public:
        //DiscontinuousGalerkin(
        //    Parameters::AllParameters *parameters_input, 
        //    Triangulation<dim>   *triangulation_input,
        //    const unsigned int degree);
        DiscontinuousGalerkin(
            Parameters::AllParameters *parameters_input, 
            const unsigned int degree);

        ~DiscontinuousGalerkin();

        void allocate_system ();
        void assemble_system ();
        double get_residual_l2norm ();

        void delete_fe_values ();

        void set_triangulation(Triangulation<dim> *triangulation_input) { triangulation = triangulation_input; } ;
        //void set_physics(Physics< dim,1,Sacado::Fad::DFad<real> > *physics) { pde_physics = physics; } ;

        /// Mesh
        Triangulation<dim>   *triangulation;

        TrilinosWrappers::SparseMatrix system_matrix;
        Vector<real> right_hand_side;
        Vector<real> newton_update;
        Vector<real> solution;

        void output_results (const unsigned int ith_grid);// const;

        /// Degrees of freedom handler allows us to iterate over the finite
        /// elements' degrees of freedom on the given triangulation
        DoFHandler<dim> dof_handler;

        /// For now, use linear mapping of domain boundaries.
        /// May need to use MappingQ or MappingQGeneric to represent curved 
        //// boundaries iso/superparametrically.
        const MappingQ<dim,dim> mapping;
        //const MappingQGeneric<dim,dim> mapping;


        // Lagrange polynomial basis
        //const FE_DGQ<dim> fe;
        /// Legendre polynomial basis
        const FE_DGP<dim> fe;

        Parameters::AllParameters *parameters;


    private:
        /// Contains the physics of the PDE
        Physics<dim, nstate, Sacado::Fad::DFad<real> >  *pde_physics;
        NumericalFluxConvective<dim, nstate, Sacado::Fad::DFad<real> > *conv_num_flux;


        void allocate_system_explicit ();

        void assemble_system_explicit ();
        void assemble_cell_terms_explicit(
            const FEValues<dim,dim> *fe_values_cell,
            const std::vector<types::global_dof_index> &current_dofs_indices,
            Vector<real> &current_cell_rhs);
        void assemble_boundary_term_explicit(
            const FEFaceValues<dim,dim> *fe_values_face_int,
            const std::vector<types::global_dof_index> &current_dofs_indices,
            Vector<real> &current_cell_rhs);
        void assemble_face_term_explicit(
            const FEValuesBase<dim,dim>     *fe_values_face_int,
            const FEFaceValues<dim,dim>     *fe_values_face_ext,
            const std::vector<types::global_dof_index> &current_dofs_indices,
            const std::vector<types::global_dof_index> &neighbor_dofs_indices,
            Vector<real>          &current_cell_rhs,
            Vector<real>          &neighbor_cell_rhs);

        void allocate_system_implicit ();
        /// Main loop of the DiscontinuousGalerkin class.
        /** It loops over all the cells, evaluates the volume contributions,
         * then loops over the faces of the current cell. Four scenarios may happen
         *
         * 1. Boundary condition.
         *
         * 2. Current face has children. Therefore, neighbor is finer. In that case,
         * loop over neighbor faces to compute its face contributions.
         *
         * 3. Neighbor has same coarseness. Cell with lower global index will be used
         * to compute the face contribution.
         *
         * 4. Neighbor is coarser. Therefore, the current cell is the finer one.
         * Do nothing since this cell will be taken care of by scenario 2.
         *    
         */
        void assemble_system_implicit ();
        void assemble_cell_terms_implicit(
            const FEValues<dim,dim> *fe_values_cell,
            const std::vector<types::global_dof_index> &current_dofs_indices,
            Vector<real> &current_cell_rhs);
        void assemble_boundary_term_implicit(
            const FEFaceValues<dim,dim> *fe_values_face_int,
            const real penalty,
            const std::vector<types::global_dof_index> &current_dofs_indices,
            Vector<real> &current_cell_rhs);
        void assemble_face_term_implicit(
            const FEValuesBase<dim,dim>     *fe_values_face_int,
            const FEFaceValues<dim,dim>     *fe_values_face_ext,
            const real penalty,
            const std::vector<types::global_dof_index> &current_dofs_indices,
            const std::vector<types::global_dof_index> &neighbor_dofs_indices,
            Vector<real>          &current_cell_rhs,
            Vector<real>          &neighbor_cell_rhs);

        // QGauss is Gauss-Legendre quadrature nodes
        const QGauss<dim>   quadrature;
        const QGauss<dim-1> face_quadrature;

        DynamicSparsityPattern sparsity_pattern;

        FEValues<dim,dim>         *fe_values_cell;
        FEFaceValues<dim,dim>     *fe_values_face_int;
        FESubfaceValues<dim,dim>  *fe_values_subface_int;
        FEFaceValues<dim,dim>     *fe_values_face_ext;

    }; // end of DiscontinuousGalerkin class

    // Returns a pointer to a DiscontinuousGalerkin object with correct template arguments
    //DiscontinuousGalerkin* create_discontinuous_galerkin (
    //    const unsigned int dimension,
    //    Parameters::AllParameters *parameters_input,
    //    const unsigned int degree,
    //    const unsigned int double_type = 0);
} // end of PHiLiP namespace

#endif
