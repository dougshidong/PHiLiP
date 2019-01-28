#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h> // Might need mapping_q
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>

namespace PHiLiP
{
    using namespace dealii;

    template <int dim, typename real>
    class PDE
    {
    public:
        PDE();
        PDE(const unsigned int polynomial_order);
        void run();

    private:
        void setup_system();
        void assemble_system();
        void solve(Vector<real> &solution);
        void output_results(const unsigned int cycle) const;

        // Mesh
        Triangulation<dim>   triangulation;

        // For now, use linear mapping of domain boundaries
        // May need to use MappingQ or MappingQGeneric to represent curved 
        // boundaries iso/superparametrically
        const MappingQ<dim> mapping;

        // Lagrange polynomial basis
        FE_DGQ<dim> fe;

        // Degrees of freedom handler allows us to iterate over the finite
        // elements' degrees of freedom on the given triangulation
        DoFHandler<dim> dof_handler;

        Vector<real> solution;
        Vector<real> right_hand_side;


        // Use MeshWorker to apply bilinear operator.
        // Main workhorse is the MeshWorker::loop function, which applies
        // a function on the cells, boundaries, and inner faces.

        // For the PDE, the bilinear form requires an inner product, 
        // which we define below

        // Use alias for simpler naming
        using DoFInfo = MeshWorker::DoFInfo<dim>;
        using CellInfo = MeshWorker::IntegrationInfo<dim>;

        static void integrate_cell_term(DoFInfo &dinfo, CellInfo &info);
        static void integrate_boundary_terms(DoFInfo &dinfo, CellInfo &info);
        static void integrate_face_terms(DoFInfo &dinfo1, 
                                         DoFInfo &dinfo2, 
                                         CellInfo &info1,
                                         CellInfo &info2);
    }; // end of PDE class
} // end of PHiLiP namespace
