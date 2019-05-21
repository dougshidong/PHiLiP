#ifndef __ADVECTION_EXPLICIT_H__
#define __ADVECTION_EXPLICIT_H__

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/mapping_q1.h> // Might need mapping_q
#include <deal.II/fe/mapping_q.h> // Might need mapping_q
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>

#include "integrator.h"

namespace PHiLiP
{
    using namespace dealii;

    template <int dim, typename real>
    class PDE
    {
    public:
        PDE();
        PDE(const unsigned int polynomial_order);
        int run();

    private:
        void compute_inv_mass_matrix();
        void compute_stiffness_matrix();

        void setup_system ();
        void setup_meshworker (IntegratorExplicit<dim,real> &integrator);

        void compute_time_step();
        void assemble_system (IntegratorExplicit<dim,real> &integrator);

        void solve(Vector<real> &solution);
        void output_results(const unsigned int cycle) const;

        // Mesh
        Triangulation<dim>   triangulation;

        // For now, use linear mapping of domain boundaries
        // May need to use MappingQ or MappingQGeneric to represent curved 
        // boundaries iso/superparametrically
        const MappingQ<dim,dim> mapping;

        // Lagrange polynomial basis
        //FE_DGQ<dim> finite_element;
        // Legendre polynomial basis
        FE_DGP<dim> fe;

        // Degrees of freedom handler allows us to iterate over the finite
        // elements' degrees of freedom on the given triangulation
        DoFHandler<dim> dof_handler;

        //// Sparse matrix needed to hold the system
        //SparsityPattern      sparsity_pattern;
        //SparseMatrix<double> system_matrix;
        //FullMatrix<real> cell_mass_matrix;
        //FullMatrix<real> cell_stiffness_matrix;
        //FullMatrix<real> cell_lifting_matrix;
        //FullMatrix<real> inverse_mass_matrix;
        //FullMatrix<real> phys_to_ref_matrix;

        Vector<real> solution;
        Vector<real> right_hand_side;
        Vector<real> source_term;

        Vector<real> cell_rhs;

        std::vector< FullMatrix<real> > inv_mass_matrix;


        // Use MeshWorker to apply bilinear operator.
        // Main workhorse is the MeshWorker::loop function, which applies
        // a function on the cells, boundaries, and inner faces.

        // For the PDE, the bilinear form requires an inner product, 
        // which we define below

        // Use alias for simpler naming
        using DoFInfo = MeshWorker::DoFInfo<dim>;
        using CellInfo = MeshWorker::IntegrationInfo<dim>;

        void integrate_cell_terms(DoFInfo &dinfo, CellInfo &info);
        void integrate_boundary_terms(DoFInfo &dinfo, CellInfo &info);
        void integrate_face_terms(DoFInfo &dinfo1, 
                                         DoFInfo &dinfo2, 
                                         CellInfo &info1,
                                         CellInfo &info2);
    }; // end of PDE class
    //template class PDE<1, double>;
    //template class PDE<2, double>;
    //template class PDE<3, double>;
} // end of PHiLiP namespace

#endif
