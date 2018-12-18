#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h> // Might need mapping_q

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

namespace PHiLiP
{
  using namespace dealii;
  // @sect3{The EulerFlow class}
  //
  // After this preparations, we proceed with the main class of this program,
  // called EulerFlow. It is basically the main class of step-6. We do
  // not have an AffineConstraints object, because there are no hanging node
  // constraints in DG discretizations.

  // Major differences will only come up in the implementation of the assemble
  // functions, since here, we not only need to cover the flux integrals over
  // faces, we also use the MeshWorker interface to simplify the loops
  // involved.
  template <int dim>
  class EulerFlow
  {
  public:
    EulerFlow();
    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve(Vector<double> &solution);
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;

    // Furthermore we want to use DG elements of degree 1 (but this is only
    // specified in the constructor). If you want to use a DG method of a
    // different degree the whole program stays the same, only replace 1 in
    // the constructor by the desired polynomial degree.
    FE_DGQ<dim>     fe;
    DoFHandler<dim> dof_handler;

    // The next four members represent the linear system to be solved.
    // <code>system_matrix</code> and <code>right_hand_side</code> are generated
    // by <code>assemble_system()</code>, the <code>solution</code> is computed
    // in <code>solve()</code>. The <code>sparsity_pattern</code> is used to
    // determine the location of nonzero elements in <code>system_matrix</code>.
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> right_hand_side;

    // Finally, we have to provide functions that assemble the cell, boundary,
    // and inner face terms. Within the MeshWorker framework, the loop over all
    // cells and much of the setup of operations will be done outside this
    // class, so all we have to provide are these three operations. They will
    // then work on intermediate objects for which first, we here define
    // alias to the info objects handed to the local integration functions
    // in order to make our life easier below.
    using DoFInfo  = MeshWorker::DoFInfo<dim>;
    using CellInfo = MeshWorker::IntegrationInfo<dim>;

    // The following three functions are then the ones that get called inside
    // the generic loop over all cells and faces. They are the ones doing the
    // actual integration.
    //
    // In our code below, these functions do not access member variables of the
    // current class, so we can mark them as <code>static</code> and simply pass
    // pointers to these functions to the MeshWorker framework. If, however,
    // these functions would want to access member variables (or needed
    // additional arguments beyond the ones specified below), we could use the
    // facilities of boost::bind (or std::bind, respectively) to provide the
    // MeshWorker framework with objects that act as if they had the required
    // number and types of arguments, but have in fact other arguments already
    // bound.
    static void integrate_cell_term(DoFInfo &dinfo, CellInfo &info);
    static void integrate_boundary_term(DoFInfo &dinfo, CellInfo &info);
    static void integrate_face_term(DoFInfo & dinfo1,
                                    DoFInfo & dinfo2,
                                    CellInfo &info1,
                                    CellInfo &info2);
  };
}

