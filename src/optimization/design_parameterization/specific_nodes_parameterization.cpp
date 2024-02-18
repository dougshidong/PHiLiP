#include "specific_nodes_parameterization.hpp"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <fstream>

namespace PHiLiP {

template<int dim>
SpecificNodesParameterization<dim> :: SpecificNodesParameterization(
    std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid)
    : BaseParameterization<dim>(_high_order_grid)
{
    store_prespecified_control_nodes();
    compute_control_index_to_vol_index();
}

template<int dim>
void SpecificNodesParameterization<dim> :: store_prespecified_control_nodes()
{
    std::ifstream infile;
    std::string filepath;
    if(this->high_order_grid->grid_degree == 1)
    {
        filepath = "q1_cylinder_controlnodes.txt";
    }
    else if (this->high_order_grid->grid_degree==2)
    {
        filepath = "q2_cylinder_controlnodes.txt";
    }
    else
    {
        std::cout<<"SpecificNodesParameterization is only implemented for q1 and q2 grids. Aborting.."<<std::endl;
        std::abort();
    }

    infile.open(filepath);
    if(!infile) {
        std::cout << "Could not open file for SpecificNodesParameteriation."<< filepath << std::endl;
        std::abort();
    }

    std::string line;
    std::getline(infile, line); // skip the first line.
    while(std::getline(infile, line))
    {
        std::stringstream ss(line);

        std::string field_x;

        std::getline(ss, field_x, ',');

        std::stringstream ss_x(field_x);
        double xval = 0.0;
        ss_x >> xval;

        x_control_nodes.push_back(xval);

        std::string field_y;

        std::getline(ss, field_y, ',');
        
        std::stringstream ss_y(field_y);

        double yval = 0;

        ss_y >> yval;

        y_control_nodes.push_back(yval);
    }
}

template<int dim>
void SpecificNodesParameterization<dim> :: compute_control_index_to_vol_index()
{
    const unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    const unsigned int n_surf_nodes = this->high_order_grid->surface_nodes.size();

    dealii::LinearAlgebra::distributed::Vector<int> is_a_control_node;
    is_a_control_node.reinit(this->high_order_grid->volume_nodes); // Copies parallel layout, without values. Initializes to 0 by default.
    is_a_control_node = 0;
    is_a_control_node.update_ghost_values();
    
    is_on_boundary.reinit(this->high_order_grid->volume_nodes); // Copies parallel layout, without values. Initializes to 0 by default.
    is_on_boundary = 0;
    is_on_boundary.update_ghost_values();
    const dealii::IndexSet &surface_range = this->high_order_grid->surface_nodes.get_partitioner()->locally_owned_range();
    for(unsigned int i_surf = 0; i_surf < n_surf_nodes; ++i_surf)
    {
        if(!(surface_range.is_element(i_surf))) continue;
        const unsigned int vol_index = this->high_order_grid->surface_to_volume_indices(i_surf);
        is_on_boundary(vol_index) = 1;
    }
    is_on_boundary.update_ghost_values();

    // Get locally owned volume and surface ranges of indices held by current processor.
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();

    for(unsigned int i_vol = 0; i_vol<n_vol_nodes; ++i_vol) 
    {
        if(!(volume_range.is_element(i_vol))) continue;
        
        if(i_vol % dim == 0.0)
        {
            if(!(volume_range.is_element(i_vol+1)))
            {
                std::cout<<"ivol+1 does not belong to the same processor, as initially expected. Aborting.."<<std::endl<<std::flush;
                std::abort();
            }
        }
        
        if(i_vol % dim == 0.0)
        {
            const double x = this->high_order_grid->volume_nodes(i_vol);
            const double y = this->high_order_grid->volume_nodes(i_vol+1);

            const bool is_part_of_region = check_if_node_belongs_to_the_region(x,y);
            
            if(is_part_of_region)
            {
                is_a_control_node(i_vol) = 1;
                is_a_control_node(i_vol+1) = 1;

                if(is_on_boundary(i_vol))
                {
                    is_a_control_node(i_vol+1) = 0;
                }
            }
        }
    }
    is_a_control_node.update_ghost_values();
     
    dealii::LinearAlgebra::distributed::Vector<int> left_control_index;
    left_control_index.reinit(this->high_order_grid->volume_nodes); // copies parallel layout, without values. initializes to 0 by default.
    left_control_index = 0;
    dealii::LinearAlgebra::distributed::Vector<int> right_control_index;
    right_control_index.reinit(this->high_order_grid->volume_nodes); // copies parallel layout, without values. initializes to 0 by default.
    right_control_index = 0;

    dealii::LinearAlgebra::distributed::Vector<int> left_update;
    left_update.reinit(this->high_order_grid->volume_nodes);
    left_update = 0;
    dealii::LinearAlgebra::distributed::Vector<int> right_update;
    right_update.reinit(this->high_order_grid->volume_nodes);
    right_update = 0;
            
    if(this->high_order_grid->get_current_fe_system().tensor_degree() > 1)
    {
    
        std::vector<int> index_adj1(16);
        std::vector<int> index_adj2(8);
        index_adj1[0] = 12;      
        index_adj1[1] = 13;
        index_adj1[2] = 12;
        index_adj1[3] = 13;
        index_adj1[4] = 8;
        index_adj1[5] = 9;
        index_adj1[6] = 14;
        index_adj1[7] = 15;
        index_adj1[8] = 16;
        index_adj1[9] = 17;
        index_adj1[10] = 16;
        index_adj1[11] = 17;
        index_adj1[12] = 16;
        index_adj1[13] = 17;
        index_adj1[14] = 16;
        index_adj1[15] = 17;
        
        index_adj2[0] = 8;
        index_adj2[1] = 9;
        index_adj2[2] = 10;
        index_adj2[3] = 11;
        index_adj2[4] = 14;
        index_adj2[5] = 15;
        index_adj2[6] = 10;
        index_adj2[7] = 11;
        

        const dealii::FESystem<dim,dim> &fe_metric = this->high_order_grid->get_current_fe_system();
        const unsigned int n_metric_dofs_cell = fe_metric.n_dofs_per_cell();
        std::vector<dealii::types::global_dof_index> dofs_indices(n_metric_dofs_cell);
        for (const auto &cell : this->high_order_grid->dof_handler_grid.active_cell_iterators()) 
        {
            if (! cell->is_locally_owned()) {continue;}

            if(this->high_order_grid->get_current_fe_system().tensor_degree() == 1) {continue;}
     
            cell->get_dof_indices (dofs_indices);

            for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof)
            {
                const unsigned int ivol = dofs_indices[idof];

                if( (ivol % dim) != 0.0) {continue;}
                
                const double x_val = this->high_order_grid->volume_nodes(ivol);

                if(is_a_control_node(ivol)==1)
                {
                    const unsigned int idof1 = index_adj1[idof];
                    const unsigned int ivol1 = dofs_indices[idof1];
                    if(is_a_control_node(ivol1)==0)
                    {
                        const double x_val1 = this->high_order_grid->volume_nodes(ivol1);

                        if(x_val1 < x_val)
                        {
                            left_control_index(ivol) += ivol1;
                            left_update(ivol) += 1;
                            if(is_a_control_node(ivol+1))
                            {
                                left_control_index(ivol+1) += ivol1+1;
                                left_update(ivol+1) += 1;
                            }
                        }
                        else
                        {
                            right_control_index(ivol) += ivol1;
                            right_update(ivol) += 1;
                            if(is_a_control_node(ivol+1))
                            {
                                right_control_index(ivol+1) += ivol1+1;
                                right_update(ivol+1) += 1;
                            }
                        }
                    } // ivol1 ends
                    if(idof <= 7)
                    {
                        const unsigned int idof2 = index_adj2[idof];
                        const unsigned int ivol2 = dofs_indices[idof2];
                        if(is_a_control_node(ivol2)==0)
                        {
                            const double x_val2 = this->high_order_grid->volume_nodes(ivol2);

                            if(x_val2 < x_val)
                            {
                                left_control_index(ivol) += ivol2;
                                left_update(ivol) += 1;
                                if(is_a_control_node(ivol+1))
                                {
                                    left_control_index(ivol+1) += ivol2+1;
                                    left_update(ivol+1) += 1;
                                }
                            }
                            else
                            {
                                right_control_index(ivol) += ivol2;
                                right_update(ivol) += 1;
                                if(is_a_control_node(ivol+1))
                                {
                                    right_control_index(ivol+1) += ivol2+1;
                                    right_update(ivol+1) += 1;
                                }
                            }
                        } // ivol2 ends
                    }
                } // ivol if ends
            } // for idof ends
        } // cell loop ends


        left_control_index.compress(dealii::VectorOperation::add);
        right_control_index.compress(dealii::VectorOperation::add);
        left_update.compress(dealii::VectorOperation::add);
        right_update.compress(dealii::VectorOperation::add);
        left_control_index.update_ghost_values();
        right_control_index.update_ghost_values();
        for(unsigned int ivol = 0; ivol < n_vol_nodes; ++ivol)
        {
            if(! volume_range.is_element(ivol)) {continue;}
            if(left_update(ivol) != 0)
            {
                left_control_index(ivol) = left_control_index(ivol)/left_update(ivol);
            }
            if(right_update(ivol) != 0)
            {
                right_control_index(ivol) = right_control_index(ivol)/right_update(ivol);
            }
        }
        left_control_index.update_ghost_values();
        right_control_index.update_ghost_values();
    } // if statement of grid degree > 1 ends.
    
    n_control_nodes = is_a_control_node.l1_norm();

    unsigned int n_control_nodes_this_processor = 0;
    for(unsigned int i_vol = 0; i_vol<n_vol_nodes; ++i_vol) 
    {
        if(!(volume_range.is_element(i_vol))) continue;
        if(is_a_control_node(i_vol) == 1) {++n_control_nodes_this_processor;}
    }


    //=========== Set inner_vol_range IndexSet of current processor ================================================================

    unsigned int n_elements_this_mpi = n_control_nodes_this_processor; // Size of local indexset
    std::vector<unsigned int> n_elements_per_mpi(this->n_mpi);
    MPI_Allgather(&n_elements_this_mpi, 1, MPI_UNSIGNED, &(n_elements_per_mpi[0]), 1, MPI_UNSIGNED, this->mpi_communicator);
    
    // Set lower index and hgher index of locally owned IndexSet on each processor
    unsigned int lower_index = 0, higher_index = 0;
    for(int i_mpi = 0; i_mpi < this->mpi_rank; ++i_mpi)
    {
        lower_index += n_elements_per_mpi[i_mpi];
    }
    higher_index = lower_index + n_elements_this_mpi;

    control_index_range.set_size(n_control_nodes);
    control_index_range.add_range(lower_index, higher_index);

    control_ghost_range.set_size(n_control_nodes);
    control_ghost_range.add_range(0,n_control_nodes);
    
    //=========== Set control_index_to_vol_index ================================================================
    control_index_to_vol_index.reinit(control_index_range, control_ghost_range, this->mpi_communicator);  
    control_index_to_left_vol_index.reinit(control_index_range, control_ghost_range, this->mpi_communicator); 
    control_index_to_right_vol_index.reinit(control_index_range, control_ghost_range, this->mpi_communicator); 

    unsigned int count1 = lower_index;
    for(unsigned int i_vol = 0; i_vol < n_vol_nodes; ++i_vol)
    {
        if(!volume_range.is_element(i_vol)) continue;
        
        if(is_a_control_node(i_vol) == 1)
        {
            if(this->high_order_grid->get_current_fe_system().tensor_degree() > 1)
            {
                control_index_to_left_vol_index[count1] = left_control_index(i_vol);
                control_index_to_right_vol_index[count1] = right_control_index(i_vol);
            }
            control_index_to_vol_index[count1++] = i_vol;
        }
    }
    AssertDimension(count1, higher_index);
    control_index_to_vol_index.update_ghost_values();
    control_index_to_left_vol_index.update_ghost_values();
    control_index_to_right_vol_index.update_ghost_values();    
}

template<int dim>
void SpecificNodesParameterization<dim> :: initialize_design_variables(VectorType &design_var)
{
    design_var.reinit(control_index_range, control_ghost_range, this->mpi_communicator);

    for(unsigned int i_control=0; i_control<n_control_nodes; ++i_control)
    {
        if(control_index_range.is_element(i_control))
        {
            const unsigned int vol_index = control_index_to_vol_index[i_control];
            design_var[i_control] = this->high_order_grid->volume_nodes[vol_index];
        }
    }
    design_var.update_ghost_values();
    current_design_var = design_var;
    current_design_var.update_ghost_values();
}

template<int dim>
void SpecificNodesParameterization<dim> :: compute_dXv_dXp(MatrixType &dXv_dXp) const
{
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(this->high_order_grid->dof_handler_grid, locally_relevant_dofs);

    const unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    
    dealii::DynamicSparsityPattern dsp(n_vol_nodes, n_control_nodes, volume_range);
    for(unsigned int i_control=0; i_control<n_control_nodes; ++i_control)
    {
        const unsigned int ivol = control_index_to_vol_index[i_control];
        if(volume_range.is_element(ivol))
        {
            dsp.add(ivol, i_control);
            if(is_on_boundary(ivol)) // Assuming only x is the control node on boundary.
            {
                dsp.add(ivol+1, i_control);
            }
        }
        
        if(this->high_order_grid->get_current_fe_system().tensor_degree() > 1)
        {
            const unsigned int ivol1 = control_index_to_left_vol_index[i_control];
            const unsigned int ivol2 = control_index_to_right_vol_index[i_control];

            if(volume_range.is_element(ivol1))
            {
                dsp.add(ivol1, i_control);
                if(is_on_boundary(ivol1))
                {
                    dsp.add(ivol1+1,i_control); 
                }
            }
            if(volume_range.is_element(ivol2))
            {
                dsp.add(ivol2, i_control);
                if(is_on_boundary(ivol2))
                {
                    dsp.add(ivol2+1,i_control); 
                }
            }
            
        }
    
    }

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, volume_range, this->mpi_communicator, locally_relevant_dofs);

    dXv_dXp.reinit(volume_range, control_index_range, dsp, this->mpi_communicator);

    for(unsigned int i_control=0; i_control<n_control_nodes; ++i_control)
    {
        const unsigned int ivol = control_index_to_vol_index[i_control];
        if(volume_range.is_element(ivol))
        {
            dXv_dXp.set(ivol, i_control, 1.0);
            if(is_on_boundary(ivol))
            {
                const bool on_boundary=true;
                const double slope = get_slope_y(ivol, on_boundary);
                dXv_dXp.set(ivol+1, i_control, slope);
            }    
        }
        
        if(this->high_order_grid->get_current_fe_system().tensor_degree() > 1)
        {
            const unsigned int ivol1 = control_index_to_left_vol_index[i_control];
            const unsigned int ivol2 = control_index_to_right_vol_index[i_control];

            if(volume_range.is_element(ivol1))
            {
                dXv_dXp.set(ivol1, i_control, 0.5);
                if(is_on_boundary(ivol1))
                {
                    const bool on_boundary=true;
                    const double half_slope = 0.5*get_slope_y(ivol1, on_boundary);
                    dXv_dXp.set(ivol1+1,i_control,half_slope);
                }
            }
            if(volume_range.is_element(ivol2))
            {
                dXv_dXp.set(ivol2, i_control, 0.5);
                if(is_on_boundary(ivol2))
                {
                    const bool on_boundary=true;
                    const double half_slope = 0.5*get_slope_y(ivol2, on_boundary);
                    dXv_dXp.set(ivol2+1,i_control,half_slope);
                }
            } 
        }
    }

    dXv_dXp.compress(dealii::VectorOperation::insert);
}

template<int dim>
bool SpecificNodesParameterization<dim> ::update_mesh_from_design_variables(
    const MatrixType &dXv_dXp,
    const VectorType &design_var)
{
    AssertDimension(dXv_dXp.n(), design_var.size());
    
    // check if design variables have changed.
    bool design_variable_has_changed = this->has_design_variable_been_updated(current_design_var, design_var);
    bool mesh_updated;
    if(!(design_variable_has_changed))
    {
        mesh_updated = false;
        return mesh_updated;
    }
    VectorType change_in_des_var = design_var;
    change_in_des_var -= current_design_var;
    change_in_des_var.update_ghost_values();

    current_design_var = design_var;
    current_design_var.update_ghost_values();
    dXv_dXp.vmult_add(this->high_order_grid->volume_nodes, change_in_des_var); // Xv = Xv + dXv_dXp*(Xp,new - Xp); Gives Xv for surface nodes and Xp,new for inner vol nodes. 
    this->high_order_grid->volume_nodes.update_ghost_values();
    mesh_updated = true;
    return mesh_updated;
}

template<int dim>
unsigned int SpecificNodesParameterization<dim> :: get_number_of_design_variables() const
{
    return n_control_nodes;
}

template<int dim>
int SpecificNodesParameterization<dim> :: is_design_variable_valid(
    const MatrixType &dXv_dXp, 
    const VectorType &design_var) const
{
    this->pcout<<"Checking if mesh is valid before updating variables..."<<std::endl;
    VectorType vol_nodes_from_design_var = this->high_order_grid->volume_nodes;
    VectorType change_in_des_var = design_var;
    change_in_des_var -= current_design_var;
    change_in_des_var.update_ghost_values();

    dXv_dXp.vmult_add(vol_nodes_from_design_var, change_in_des_var); // Xv = Xv + dXv_dXp*(Xp,new - Xp); Gives Xv for surface nodes and Xp,new for inner vol nodes. 
    vol_nodes_from_design_var.update_ghost_values();
    
    int mesh_error_this_processor = 0;
    const dealii::FESystem<dim,dim> &fe_metric = this->high_order_grid->get_current_fe_system();
    const unsigned int n_dofs_per_cell = fe_metric.n_dofs_per_cell();
    const std::vector< dealii::Point<dim> > &ref_points = fe_metric.get_unit_support_points();
    for (const auto &cell : this->high_order_grid->dof_handler_grid.active_cell_iterators()) 
    {
        if (! cell->is_locally_owned()) {continue;}

        const std::vector<double> jac_det = this->high_order_grid->evaluate_jacobian_at_points(vol_nodes_from_design_var, cell, ref_points);
        for (unsigned int i=0; i<n_dofs_per_cell; ++i) 
        {
            if(jac_det[i] < 1.0e-12)
            {
                std::cout<<"Cell is distorted"<<std::endl;
                ++mesh_error_this_processor;
                break;
            }
        }

        if(mesh_error_this_processor > 0) {break;}
    }

    const int mesh_error_mpi = dealii::Utilities::MPI::sum(mesh_error_this_processor, this->mpi_communicator);
    return mesh_error_mpi;
}
    
template<int dim>
bool SpecificNodesParameterization<dim> :: check_if_node_belongs_to_the_region(const double x, const double y) const
{
    const double tol = 1.0e-5;
    for(unsigned int i=0; i<x_control_nodes.size(); ++i)
    {
        if( sqrt(pow(x-x_control_nodes[i],2) + pow(y-y_control_nodes[i],2)) < tol )
        {
            return true;
        }
    }
    return false;
}

template<int dim>
double SpecificNodesParameterization<dim> :: get_slope_y(
    const unsigned int ivol, 
    const bool on_boundary) const
{
    if(! on_boundary) {return 0.0;}
   
    const double PI = 4.0*atan(1.0);
    double slope = 5.0/2.0 * tan(5.0*PI/12.0);
    if(this->high_order_grid->volume_nodes(ivol + 1) > 0)
    {
        slope = -slope;
    }

    return slope;
}

template<int dim>
std::vector<std::pair<double,double>> SpecificNodesParameterization<dim> :: get_final_control_nodes_list() const
{
    this->high_order_grid->volume_nodes.update_ghost_values();
    const unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    dealii::IndexSet ghost_index_serial;
    ghost_index_serial.set_size(n_vol_nodes);
    ghost_index_serial.add_range(0,n_vol_nodes);
    const dealii::IndexSet &local_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    VectorType volume_nodes_serial;
    volume_nodes_serial.reinit(local_range, ghost_index_serial, this->mpi_communicator);
    volume_nodes_serial = this->high_order_grid->volume_nodes;
    volume_nodes_serial.update_ghost_values();
        
    std::vector<std::pair<double,double>> final_control_nodes_list;
    if(this->mpi_rank == 0)
    {
        std::vector<double> xvals, yvals;

        for(unsigned int icontrol=0; icontrol<n_control_nodes; ++icontrol)
        {
            const unsigned int ivol = control_index_to_vol_index[icontrol];
            if(ivol % dim ==0)
            {
                const double xval = volume_nodes_serial(ivol);
                const double yval = volume_nodes_serial(ivol+1);
                final_control_nodes_list.push_back(std::make_pair(xval, yval));
            } 
        }
        // sort control nodes list in ascending order of y values.
        std::sort(final_control_nodes_list.begin(), 
                  final_control_nodes_list.end(), 
                  [](const std::pair<double,double> &left, const std::pair<double,double> &right){return left.second < right.second;});
    } // mpi rank = 0
    return final_control_nodes_list;
}

template<int dim>
void SpecificNodesParameterization<dim> :: write_control_nodes_to_file(
    const std::vector<std::pair<double,double>> &final_control_nodes_list) const
{
    if(this->mpi_rank == 0)
    {
        std::ofstream outfile("q2_cylinder_controlnodes.txt"); 

        if(! outfile.is_open())
        {
            std::cout<<"Could not open the file. Aborting.."<<std::endl;
            std::abort();
        }

        outfile<<"x,y"<<"\n";

        // Write control nodes
        for(unsigned int inode = 0; inode<final_control_nodes_list.size(); ++inode)
        {
            outfile<<final_control_nodes_list[inode].first<<","<<final_control_nodes_list[inode].second<<"\n";
        }
        outfile.close();
    } 
}

template<int dim>
void SpecificNodesParameterization<dim> :: output_control_nodes(const std::string /*filename*/) const
{
    std::vector<std::pair<double,double>> final_control_nodes_list = get_final_control_nodes_list();
    write_control_nodes_to_file(final_control_nodes_list);
}

template<int dim>
void SpecificNodesParameterization<dim> :: output_control_nodes_with_interpolated_high_order_nodes() const
{
    std::vector<std::pair<double,double>> final_control_nodes_list = get_final_control_nodes_list();
    AssertDimension(final_control_nodes_list.size(), n_control_nodes);
    if(this->mpi_rank == 0)
    {
        // Compute high-order q2 nodes at center
        for(unsigned int icontrol=0; icontrol<n_control_nodes-1; ++icontrol)
        {
            const double xval_i = final_control_nodes_list[icontrol].first;
            const double xval_ip1 = final_control_nodes_list[icontrol+1].first;
            const double yval_i = final_control_nodes_list[icontrol].second;
            const double yval_ip1 = final_control_nodes_list[icontrol+1].second;
            const double xval_mid = (xval_ip1 + xval_i)/2.0;
            const double yval_mid = (yval_ip1 + yval_i)/2.0;
            final_control_nodes_list.push_back(std::make_pair(xval_mid, yval_mid));
        }
                
        std::sort(final_control_nodes_list.begin(), final_control_nodes_list.end(), [](const std::pair<double,double> &left, const std::pair<double,double> &right) {
        return left.second < right.second;
        });
    }

    write_control_nodes_to_file(final_control_nodes_list); 
}

template<int dim>
void SpecificNodesParameterization<dim> :: output_control_nodes_refined() const
{
    std::vector<std::pair<double,double>> final_control_nodes_list = get_final_control_nodes_list();
    AssertDimension(final_control_nodes_list.size(), n_control_nodes);
    AssertDimension(this->high_order_grid->grid_degree,2);
    if(this->mpi_rank == 0)
    {
        // Compute high-order q2 nodes at center
        for(unsigned int icontrol=0; icontrol<n_control_nodes-2; icontrol+=2)
        {
            const double x1 = final_control_nodes_list[icontrol].first;
            const double y1 = final_control_nodes_list[icontrol].second;
            const double x2 = final_control_nodes_list[icontrol+1].first;
            const double y2 = final_control_nodes_list[icontrol+1].second;
            const double x3 = final_control_nodes_list[icontrol+2].first;
            const double y3 = final_control_nodes_list[icontrol+2].second;
            const double bx = 4*x2 - x3 - 3*x1;
            const double ax = x3-x1-bx;
            const double by = 4*y2 - y3 - 3*y1;
            const double ay = y3-y1-by;
            const double cx = x1;
            const double cy = y1;
            const double x12 = ax/16.0 + bx/4.0 + cx;
            const double y12 = ay/16.0 + by/4.0 + cy;
            const double x23 = ax*9.0/16.0 + bx*3.0/4.0 + cx;
            const double y23 = ay*9.0/16.0 + by*3.0/4.0 + cy;
            final_control_nodes_list.push_back(std::make_pair(x12, y12));
            final_control_nodes_list.push_back(std::make_pair(x23, y23));
        }
                
        std::sort(final_control_nodes_list.begin(), final_control_nodes_list.end(), [](const std::pair<double,double> &left, const std::pair<double,double> &right) {
        return left.second < right.second;
        });
    }

    write_control_nodes_to_file(final_control_nodes_list);     
}

template class SpecificNodesParameterization<PHILIP_DIM>;
} // namespace PHiLiP
