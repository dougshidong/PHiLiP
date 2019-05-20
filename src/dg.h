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
#include "numerical_flux/numerical_flux.h"
#include "parameters.h"


namespace PHiLiP
{
    using namespace dealii;

    /// DG is independent of the number of state variables.
    /**  This base class allows the use of arrays to efficiently allocate the data structures
      *  through std::array in the derived class DG.
      *  This class is the one being returned by the DGFactory and is the main
      *  interface for a user to call its main functions such as "assemble_system"
      */
    template <int dim, typename real>
    class DGBase 
    {
    public:

        DGBase () = delete; // Can't initialize this without inputs
        DGBase(
            Parameters::AllParameters *parameters_input, 
            const unsigned int degree);

        virtual ~DGBase();

        virtual void allocate_system () = 0;
        virtual void assemble_system () = 0;
        void evaluate_inverse_mass_matrices ();
        double get_residual_l2norm ();

        void delete_fe_values ();

        void set_triangulation(Triangulation<dim> *triangulation_input) { triangulation = triangulation_input; } ;
        //void set_physics(Physics< dim,1,Sacado::Fad::DFad<real> > *physics) { pde_physics = physics; } ;

        /// Mesh
        Triangulation<dim>   *triangulation;


        std::vector<FullMatrix<real>> inv_mass_matrix;

        DynamicSparsityPattern sparsity_pattern;
        TrilinosWrappers::SparseMatrix system_matrix;
        Vector<real> right_hand_side;
        Vector<real> newton_update;
        Vector<real> solution;

        void output_results (const unsigned int ith_grid);// const;

        const MappingQ<dim> mapping;

        /// Lagrange polynomial basis
        const FE_DGQ<dim> fe;
        //const FE_DGQLegendre<dim> fe;

        Parameters::AllParameters *parameters;


        /// Degrees of freedom handler allows us to iterate over the finite
        /// elements' degrees of freedom on the given triangulation
        /// Must be defined after fe since it is a subscriptor of fe.
        /// Destructor are called in reverse order in which they appear in class definition. 
        DoFHandler<dim> dof_handler;


    protected:

        // QGauss is Gauss-Legendre quadrature nodes
        const QGauss<dim>   quadrature;
        const QGauss<dim-1> face_quadrature;

        FEValues<dim,dim>         *fe_values_cell;
        FEFaceValues<dim,dim>     *fe_values_face_int;
        FESubfaceValues<dim,dim>  *fe_values_subface_int;
        FEFaceValues<dim,dim>     *fe_values_face_ext;
        /// Main loop of the DGBase class.
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
        //virtual void allocate_system_implicit () = 0;
        //virtual void assemble_system_implicit () = 0;

    }; // end of DG class

    template <int dim, int nstate, typename real>
    class DG : public DGBase<dim, real>
    {
    public:
        DG(
            Parameters::AllParameters *parameters_input, 
            const unsigned int degree);

        ~DG();

    private:
        /// Contains the physics of the PDE
        Physics<dim, nstate, Sacado::Fad::DFad<real> >  *pde_physics;
        NumericalFluxConvective<dim, nstate, Sacado::Fad::DFad<real> > *conv_num_flux;
        NumericalFluxDissipative<dim, nstate, Sacado::Fad::DFad<real> > *diss_num_flux;


        void allocate_system ();
        /// Main loop of the DG class.
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
        void assemble_system ();
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

    }; // end of DG class

    // Returns a pointer to a DG object with correct template arguments
    //DG* create_discontinuous_galerkin (
    //    const unsigned int dimension,
    //    Parameters::AllParameters *parameters_input,
    //    const unsigned int degree,
    //    const unsigned int double_type = 0);

    /// This class creates a new DGBase object
    /** This allows the DGBase to not be templated on the number of state variables
      * while allowing DG to be template on the number of state variables
     */
    template <int dim, typename real>
    class DGFactory
    {
    public:
        //static DGBase<dim,real>*
        static std::shared_ptr< DGBase<dim,real> >
            create_discontinuous_galerkin(
            Parameters::AllParameters *parameters, 
            const unsigned int degree);
    };
} // end of PHiLiP namespace

#endif
