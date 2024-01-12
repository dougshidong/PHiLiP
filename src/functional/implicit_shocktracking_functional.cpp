#include "implicit_shocktracking_functional.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include "linear_solver/linear_solver.h"
#include "mesh_jacobian_deviation_functional.h"
#include "cell_distortion_functional.h"

namespace PHiLiP {

template <int dim, int nstate, typename real>
ImplicitShockTrackingFunctional<dim, nstate, real> :: ImplicitShockTrackingFunctional( 
    std::shared_ptr<DGBase<dim,real>> dg_input,
    const bool uses_solution_values,
    const bool uses_solution_gradient,
    const bool _use_coarse_residual)
    : Functional<dim, nstate, real> (dg_input, uses_solution_values, uses_solution_gradient)
    , use_coarse_residual(_use_coarse_residual)
    , mesh_weight(this->dg->all_parameters->optimization_param.mesh_weight_factor)
    , initial_vol_nodes(this->dg->high_order_grid->volume_nodes)
    , coarse_poly_degree(this->dg->get_min_fe_degree())
    , fine_poly_degree(coarse_poly_degree + 1)
    , use_gauss_newton(false)
{
    if(this->dg->get_min_fe_degree() != this->dg->get_max_fe_degree())
    {
        std::cout<<"This class is currently coded assuming a constant poly degree. To be changed in future if required."<<std::endl;
        std::abort();
    }
    compute_interpolation_matrix(); // also stores cellwise_dofs_fine, vector coarse and vector fine.
    if(use_coarse_residual)
    {
        this->pcout<<"Using coarse residual."<<std::endl;
    }
    
    //cell_distortion_functional = std::make_unique<MeshJacobianDeviation<dim, nstate, real>> (this->dg);
    cell_distortion_functional = std::make_unique<CellDistortion<dim, nstate, real>> (this->dg);
}

//===================================================================================================================================================
//                          Functions used only once in constructor
//===================================================================================================================================================
template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: compute_interpolation_matrix()
{ 
    vector_coarse = this->dg->solution; // copies values and parallel layout
    vector_vol_nodes = this->dg->high_order_grid->volume_nodes;
    unsigned int n_dofs_coarse = this->dg->n_dofs();
    this->dg->set_p_degree_and_interpolate_solution(fine_poly_degree);
    vector_fine = this->dg->solution;
    unsigned int n_dofs_fine = this->dg->n_dofs();
    const dealii::IndexSet dofs_fine_locally_relevant_range = this->dg->locally_relevant_dofs;
    cellwise_dofs_fine = get_cellwise_dof_indices();

    this->dg->set_p_degree_and_interpolate_solution(coarse_poly_degree);
    AssertDimension(vector_coarse.size(), this->dg->solution.size());     

    // Get all possible interpolation matrices for available poly order combinations.
    dealii::Table<2,dealii::FullMatrix<real>> interpolation_hp;
    extract_interpolation_matrices(interpolation_hp);

    // Get locally owned dofs
    const dealii::IndexSet &dofs_range_coarse = vector_coarse.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &dofs_range_fine = vector_fine.get_partitioner()->locally_owned_range();

    dealii::DynamicSparsityPattern dsp(n_dofs_fine, n_dofs_coarse, dofs_range_fine);
    std::vector<dealii::types::global_dof_index> dof_indices;
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        for(unsigned int i=0; i < dof_indices_fine.size(); ++i)
        {
            for(unsigned int j=0; j < n_dofs_cell; ++j)
            {
                dsp.add(dof_indices_fine[i], dof_indices[j]);
            }
        }

    } // cell loop ends

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dofs_range_fine, MPI_COMM_WORLD, dofs_fine_locally_relevant_range);
    interpolation_matrix.reinit(dofs_range_fine, dofs_range_coarse, dsp, MPI_COMM_WORLD);

    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];
        
        assert(i_fele + 1 <= this->dg->max_degree);
        const dealii::FullMatrix<real> &interpolation_matrix_local = interpolation_hp(i_fele + 1, i_fele);
        AssertDimension(interpolation_matrix_local.m(), dof_indices_fine.size());
        AssertDimension(interpolation_matrix_local.n(), n_dofs_cell);

        for(unsigned int i=0; i < dof_indices_fine.size(); ++i)
        {
            for(unsigned int j=0; j < n_dofs_cell; ++j)
            {
                interpolation_matrix.set(dof_indices_fine[i], dof_indices[j], interpolation_matrix_local(i,j));
            }
        }

    } // cell loop ends

    interpolation_matrix.compress(dealii::VectorOperation::insert);
}

template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: extract_interpolation_matrices(
    dealii::Table<2, dealii::FullMatrix<real>> &interpolation_hp)
{
    const dealii::hp::FECollection<dim> &fe = this->dg->dof_handler.get_fe_collection();
    interpolation_hp.reinit(fe.size(), fe.size());

    for(unsigned int i=0; i<fe.size(); ++i)
    {
        for(unsigned int j=0; j<fe.size(); ++j)
        {
            if(i != j)
            {
                interpolation_hp(i, j).reinit(fe[i].n_dofs_per_cell(), fe[j].n_dofs_per_cell());
                try
                {
                    fe[i].get_interpolation_matrix(fe[j], interpolation_hp(i,j));
                    //get_projection_matrix(fe[i], fe[j], interpolation_hp(i,j));
                } 
                // If interpolation matrix cannot be generated, reset matrix size to 0.
                catch (const typename dealii::FiniteElement<dim>::ExcInterpolationNotImplemented &)
                {
                    interpolation_hp(i,j).reinit(0,0);
                }

            }
        }
    }
}

template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: get_projection_matrix(
    const dealii::FESystem<dim,dim> &fe_i, // fe output
    const dealii::FESystem<dim,dim> &fe_j, // fe input
    dealii::FullMatrix<real> &projection_matrix)
{
    const unsigned int degree = std::max(fe_i.tensor_degree(), fe_j.tensor_degree());
    const dealii::QGauss<dim> projection_quadrature(degree+5);
    const unsigned int in_vector_size = projection_matrix.n();
    const unsigned int out_vector_size = projection_matrix.m();
    for(unsigned int i=0; i<in_vector_size; ++i)
    {
        std::vector<real> identity_column (in_vector_size);
        std::fill(identity_column.begin(), identity_column.end(), 0.0);
        identity_column[i] = 1.0;

        const std::vector<real> out_vector = project_function<dim, real>(identity_column, fe_j, fe_i, projection_quadrature);
        AssertDimension(out_vector.size(), out_vector_size); 

        for(unsigned int row = 0; row < out_vector_size; ++row)
        {
            projection_matrix(row, i) = out_vector[row];
        }
    } // for ends
}

template<int dim, int nstate, typename real>
std::vector<std::vector<dealii::types::global_dof_index>> ImplicitShockTrackingFunctional<dim, nstate, real> :: get_cellwise_dof_indices()
{
    std::vector<std::vector<dealii::types::global_dof_index>> cellwise_dof_indices(this->dg->triangulation->n_active_cells());
    std::vector<dealii::types::global_dof_index> dof_indices;
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        cellwise_dof_indices[cell_index] = dof_indices;
    }

    return cellwise_dof_indices;
}

//===================================================================================================================================================
//                          Functions used in evaluate_functional
//===================================================================================================================================================

template<int dim, int nstate, typename real>
real ImplicitShockTrackingFunctional<dim, nstate, real> :: evaluate_functional(
    const bool compute_dIdW,
    const bool compute_dIdX,
    const bool compute_d2I)
{
    bool actually_compute_value = true;
    bool actually_compute_dIdW = compute_dIdW;
    bool actually_compute_dIdX = compute_dIdX;
    bool actually_compute_d2I  = compute_d2I;


    if(compute_dIdW || compute_dIdX || compute_d2I)
    {
        actually_compute_dIdW = true;
        actually_compute_dIdX = true;
        actually_compute_d2I  = false; // Hessian-vector products are evaluated directly without storing d2Is. 
    }

    this->need_compute(actually_compute_value, actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);
    
    bool compute_derivatives = false;
    if(actually_compute_dIdW || actually_compute_dIdX) {compute_derivatives = true;}

    if(actually_compute_value)
    {
        this->current_functional_value = evaluate_objective_function(); // also stores adjoint and residual_used.
        this->pcout<<"Evaluated objective function."<<std::endl;
        AssertDimension(this->dg->solution.size(), vector_coarse.size());
    }

    if(compute_derivatives)
    {
        this->pcout<<"Computing common vectors and matrices."<<std::endl;
        compute_common_vectors_and_matrices();
        AssertDimension(this->dg->solution.size(), vector_coarse.size());
        this->pcout<<"Computed common vectors and matrices."<<std::endl;
        store_dIdX();
        this->pcout<<"Stored dIdX."<<std::endl;
        store_dIdW();
        this->pcout<<"Stored dIdw."<<std::endl;
    }

    return this->current_functional_value;
}


template<int dim, int nstate, typename real>
real ImplicitShockTrackingFunctional<dim, nstate, real> :: evaluate_objective_function()
{
    const VectorType solution_coarse_stored = this->dg->solution;
    this->dg->set_p_degree_and_interpolate_solution(fine_poly_degree);
    this->dg->assemble_residual();
    residual_fine = this->dg->right_hand_side;
    residual_fine.update_ghost_values();
    const real obj_func_val = 0.5* (residual_fine * residual_fine);
    
    this->dg->set_p_degree_and_interpolate_solution(coarse_poly_degree);
    /* Interpolating one poly order up and then down changes solution by ~1.0e-12, which causes functional to be re-evaluated when the solution-node configuration is the same. 
    Resetting of solution to stored coarse solution prevents this issue.     */
    this->dg->solution = solution_coarse_stored; 
    this->dg->solution.update_ghost_values();
    
    const real obj_func_net = obj_func_val + cell_distortion_functional->evaluate_functional();
    return obj_func_net;
}

template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: compute_common_vectors_and_matrices()
{
    const VectorType solution_coarse_stored = this->dg->solution;
    this->dg->set_p_degree_and_interpolate_solution(fine_poly_degree);
    
    // Store derivatives related to the residual
    bool compute_dRdW = true, compute_dRdX=false, compute_d2R=false;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    R_u.reinit(this->dg->system_matrix);
    R_u.copy_from(this->dg->system_matrix);
    
    compute_dRdW = false, compute_dRdX = true, compute_d2R = false;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    R_x.reinit(this->dg->dRdXv);
    R_x.copy_from(this->dg->dRdXv);
 
    AssertDimension(residual_fine.size(), vector_fine.size());
    AssertDimension(residual_fine.size(), this->dg->solution.size());
    this->dg->set_dual(residual_fine);
    compute_dRdW = false, compute_dRdX = false, compute_d2R = true;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    R_times_Rux.reinit(this->dg->d2RdWdX);
    R_times_Rux.copy_from(this->dg->d2RdWdX);
    R_times_Ruu.reinit(this->dg->d2RdWdW);
    R_times_Ruu.copy_from(this->dg->d2RdWdW);
    R_times_Rxx.reinit(this->dg->d2RdXdX);
    R_times_Rxx.copy_from(this->dg->d2RdXdX);

    this->dg->set_p_degree_and_interpolate_solution(coarse_poly_degree);
    
    /* Interpolating one poly order up and then down changes solution by ~1.0e-12, which causes functional to be re-evaluated when the solution-node configuration is the same. 
    Resetting of solution to stored coarse solution prevents this issue.     */
    this->dg->solution = solution_coarse_stored; 
    this->dg->solution.update_ghost_values();
    
    // Compress all matrices
    R_u.compress(dealii::VectorOperation::add);
    R_x.compress(dealii::VectorOperation::add);
    R_times_Rux.compress(dealii::VectorOperation::add);
    R_times_Ruu.compress(dealii::VectorOperation::add);
    R_times_Rxx.compress(dealii::VectorOperation::add);
}

template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: store_dIdX()
{ 
    this->dIdX.reinit(vector_vol_nodes);
    R_x.Tvmult(this->dIdX, residual_fine);
    this->dIdX.update_ghost_values();


    // Add derivative of cell distortion measure
    const bool compute_dIdW = false, compute_dIdX = true, compute_d2I = false;
    cell_distortion_functional->evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
    this->dIdX += cell_distortion_functional->dIdX;
    this->dIdX.update_ghost_values();
}

template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: store_dIdW()
{
    this->dIdw.reinit(vector_coarse);
    
    VectorType fine_R_times_Ru(vector_fine);
    R_u.Tvmult(fine_R_times_Ru, residual_fine);
    fine_R_times_Ru.update_ghost_values();
    interpolation_matrix.Tvmult(this->dIdw, fine_R_times_Ru);
    this->dIdw.update_ghost_values();
}

template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: d2IdWdW_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), vector_coarse.size());

    VectorType in_vector_fine(vector_fine);
    interpolation_matrix.vmult(in_vector_fine, in_vector);
    in_vector_fine.update_ghost_values();
    //========= Evaluate term1 =================
    VectorType term1(vector_fine);
    if(use_gauss_newton)
    {
        term1 = 0.0;
    }
    else
    {
        R_times_Ruu.vmult(term1, in_vector_fine);
    }
    term1.update_ghost_values();

    //========= Evaluate term2 =================
    VectorType term_intermediate(vector_fine);
    R_u.vmult(term_intermediate, in_vector_fine);
    term_intermediate.update_ghost_values();

    VectorType term2(vector_fine);
    R_u.Tvmult(term2, term_intermediate);
    term2.update_ghost_values();

    // ======================================

    VectorType out_vector_fine = term1;
    out_vector_fine += term2;
    out_vector_fine.update_ghost_values();

    interpolation_matrix.Tvmult(out_vector, out_vector_fine);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: d2IdWdX_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{ 
    AssertDimension(in_vector.size(), this->dg->high_order_grid->volume_nodes.size());
    AssertDimension(out_vector.size(), vector_coarse.size());

    //========= Evaluate term1 =================
    VectorType term1(vector_fine);
    if(use_gauss_newton)
    {
        term1 = 0.0;
    }
    else
    {
        R_times_Rux.vmult(term1, in_vector);
    }
    term1.update_ghost_values();

    //========= Evaluate term2 =================
    VectorType term_intermediate(vector_fine);
    R_x.vmult(term_intermediate, in_vector);
    term_intermediate.update_ghost_values();

    VectorType term2(vector_fine);
    R_u.Tvmult(term2, term_intermediate);
    term2.update_ghost_values();
    // ======================================

    VectorType out_vector_fine = term1;
    out_vector_fine += term2;
    out_vector_fine.update_ghost_values();

    interpolation_matrix.Tvmult(out_vector, out_vector_fine);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: d2IdWdX_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{ 
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), this->dg->high_order_grid->volume_nodes.size());

    VectorType in_vector_fine(vector_fine);
    interpolation_matrix.vmult(in_vector_fine, in_vector);
    in_vector_fine.update_ghost_values();

    // ========= Evaluate term1 ==========================
    VectorType term_intermediate(vector_fine);
    R_u.vmult(term_intermediate, in_vector_fine);
    term_intermediate.update_ghost_values();

    VectorType term1(vector_vol_nodes);
    R_x.Tvmult(term1, term_intermediate);
    term1.update_ghost_values();

    // ========= Evaluate term2 ==========================
    VectorType term2(vector_vol_nodes);
    if(use_gauss_newton)
    {
        term2 = 0.0;
    }
    else
    {
        R_times_Rux.Tvmult(term2, in_vector_fine);
    }
    term2.update_ghost_values();

    //====================================================
    out_vector = term1;
    out_vector += term2;
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void ImplicitShockTrackingFunctional<dim, nstate, real> :: d2IdXdX_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), this->dg->high_order_grid->volume_nodes.size());
    AssertDimension(out_vector.size(), this->dg->high_order_grid->volume_nodes.size());
    
    //========= Evaluate term1 =================
    VectorType term1(vector_vol_nodes);
    if(use_gauss_newton)
    {
        term1 = 0.0;
    }
    else
    {
        R_times_Rxx.vmult(term1, in_vector);
    }
    term1.update_ghost_values();

    //========= Evaluate term2 =================
    VectorType term_intermediate(vector_fine);
    R_x.vmult(term_intermediate, in_vector);
    term_intermediate.update_ghost_values();

    VectorType term2(vector_vol_nodes);
    R_x.Tvmult(term2, term_intermediate);
    term2.update_ghost_values();

    // ======================================
    out_vector = term1;
    out_vector += term2;
    out_vector.update_ghost_values();


    // Add Hessian-vector product of mesh distortion weight
    const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
    cell_distortion_functional->evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
    VectorType out_vector2(out_vector);
    cell_distortion_functional->d2IdXdX_vmult(out_vector2, in_vector);
    out_vector2.update_ghost_values();

    out_vector += out_vector2;
    out_vector.update_ghost_values();
}

template class ImplicitShockTrackingFunctional <PHILIP_DIM, 1, double>;
template class ImplicitShockTrackingFunctional <PHILIP_DIM, 2, double>;
template class ImplicitShockTrackingFunctional <PHILIP_DIM, 3, double>;
template class ImplicitShockTrackingFunctional <PHILIP_DIM, 4, double>;
template class ImplicitShockTrackingFunctional <PHILIP_DIM, 5, double>;
} // namespace PHiLiP

