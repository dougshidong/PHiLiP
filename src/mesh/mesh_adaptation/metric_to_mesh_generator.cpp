#include "metric_to_mesh_generator.h"
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/distributed/shared_tria.h>
#include <ostream>
#include <fstream>
#include "grid_refinement/gmsh_out.h"

namespace PHiLiP {

template<int dim, int nstate, typename real, typename MeshType>
MetricToMeshGenerator<dim, nstate, real, MeshType> :: MetricToMeshGenerator(
	std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> _volume_nodes_mapping,
	std::shared_ptr<MeshType> _triangulation)
	: volume_nodes_mapping(_volume_nodes_mapping)
	, triangulation(_triangulation)
	, dof_handler_vertices(*triangulation)
	, fe_q(1)
	, fe_system(dealii::FESystem<dim>(fe_q,1)) // setting nstate = 1 to index vertices
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
	, filename("mesh_to_be_generated")
	, filename_pos(filename + ".pos")
	, filename_geo(filename + ".geo")
{
    MPI_Comm_rank(mpi_communicator, &mpi_rank);
    MPI_Comm_size(mpi_communicator, &n_mpi);

	if(n_mpi != 1)
	{
		pcout<<"Error: MetricToMeshGenerator currently works with only 1 processor. Need to update it in future."<<std::flush;
		std::abort();
	}

	reinit();
}

template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: reinit()
{
	dof_handler_vertices.initialize(*triangulation, fe_system);
	dof_handler_vertices.distribute_dofs(fe_system);
	dealii::DoFRenumbering::Cuthill_McKee(dof_handler_vertices);

	const unsigned int n_vertices = dof_handler_vertices.n_dofs();
	
	optimal_metric_at_vertices.clear();
	
    dealii::Tensor<2, dim, real> zero_tensor; // initialized to 0 by default.
	for(unsigned int i=0; i<n_vertices; ++i)
	{
		optimal_metric_at_vertices.push_back(zero_tensor);
	}

	all_vertices.clear();
	all_vertices.resize(n_vertices);
}
	
template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: interpolate_metric_to_vertices(
	const std::vector<dealii::Tensor<2, dim, real>> &cellwise_optimal_metric)
{
	const unsigned int n_vertices = dof_handler_vertices.n_dofs();

	dealii::Quadrature<dim> quadrature = fe_system.get_unit_support_points();
	dealii::FEValues<dim, dim> fe_values_vertices(*volume_nodes_mapping, fe_system, quadrature, dealii::update_quadrature_points);
	const unsigned int n_dofs_cell = fe_system.dofs_per_cell;
	std::vector<dealii::types::global_dof_index> dof_indices(n_dofs_cell);

	// Store vertices in each cell with global dof.
	for(const auto &cell: dof_handler_vertices.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}

		fe_values_vertices.reinit(cell);
		cell->get_dof_indices(dof_indices);

		const std::vector<dealii::Point<dim>> &cell_vertices = fe_values_vertices.get_quadrature_points();

		for(unsigned int idof = 0; idof < cell_vertices.size(); ++idof)
		{
			const unsigned int iquad = idof;
			all_vertices[dof_indices[idof]] = cell_vertices[iquad];
		}
		
		if(cell_vertices.size() != dealii::GeometryInfo<dim>::vertices_per_cell) 
		{
			std::cout<<"Size of cell_vertices needs to be equal to vertices per cell. Aborting.."<<std::flush;
			std::abort();
		}
		// Check if vertices are in the same order.
		for(unsigned int idof = 0; idof < cell_vertices.size(); ++idof)
		{
			const dealii::Point<dim> expected_vertex = cell->vertex(idof);
			const real error_in_vertex = expected_vertex.distance(all_vertices[dof_indices[idof]]);
			//std::cout<<"Expected vertex = "<<cell->vertex(idof)<<"     "<<"Actual vertex = "<<all_vertices[dof_indices[idof]]<<std::endl;
			if(error_in_vertex > 1.0e-10)
			{
				std::cout<<"Error in ordering of vertices. Aborting.."<<std::flush;
				std::abort();
			}	
		}
	} // cell loop ends

//================================ Interpolate metric field ================================================

	std::vector<int> metric_count_at_vertices; // initialized to 0
	metric_count_at_vertices.resize(n_vertices);

	for(const auto &cell: dof_handler_vertices.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}

		cell->get_dof_indices(dof_indices);
		const unsigned int cell_index = cell->active_cell_index();
		
		for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
		{
			const unsigned int idof_global = dof_indices[idof];
			optimal_metric_at_vertices[idof_global] += cellwise_optimal_metric[cell_index];
			++metric_count_at_vertices[idof_global]; 
		}

	} // cell loop ends
	
    // Compute average
    for(unsigned int i=0; i<n_vertices; ++i)
    {
        optimal_metric_at_vertices[i] /= metric_count_at_vertices[i];
    }

	// Output optimal metric at vertices for verifying.
	pcout<<"Optimal metric at vertices = "<<std::endl;
    for(const auto &cell : dof_handler_vertices.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}
		cell->get_dof_indices(dof_indices);
		
		std::cout<<"Cell index = "<<cell->active_cell_index()<<std::endl;
		std::cout<<"Vertices in this cell = "<<std::endl;
		for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
		{
			const unsigned int idof_global = dof_indices[idof];
			std::cout<<"Vertex = "<<all_vertices[idof_global]<<std::endl;
			std::cout<<"metric count = "<<metric_count_at_vertices[idof_global]<<std::endl;
		}
		std::cout<<"Metric at vertex = "<<std::endl;
	
		for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
		{
			const unsigned int idof_global = dof_indices[idof];
			std::cout<<"Metric at vertex "<<all_vertices[idof_global]<<" :"<<std::endl;
        
			for(unsigned int i = 0; i<dim; ++i)
			{
				for(unsigned int j=0; j<dim; ++j)
				{
					std::cout<<optimal_metric_at_vertices[idof_global][i][j]<<" ";
				}
				std::cout<<std::endl;
			}
		}

		std::cout<<"Metric at cell = "<<std::endl;
		for(unsigned int i = 0; i<dim; ++i)
		{
			for(unsigned int j=0; j<dim; ++j)
			{
				std::cout<<cellwise_optimal_metric[cell->active_cell_index()][i][j]<<" ";
			}
			std::cout<<std::endl;
		}
		
		std::cout<<std::endl;
	
    } // cell loop ends
	
}

template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: write_pos_file()
{
	if(dim != 2)
	{
		pcout<<"This function is currently hardcoded for dim = 2. Might change it later when required."<<std::endl<<std::flush;
		std::abort();
	}
	AssertDimension(fe_system.dofs_per_cell, dealii::GeometryInfo<dim>::vertices_per_cell);
	
	// Based on gmsh/tutorials/t17_bgmesh.pos.
	// Adapted from GridRefinement::Gmsh_Out::write_pos_anisotropic() to use metric field at nodes.
	const std::string quotes = "\"";
	std::ofstream outfile(filename_pos);
	outfile<<"// Background mesh with containing optimal metric field."<<'\n'; 
	outfile<< "View "<<quotes<<"nodalMetric"<<quotes<<" {"<<'\n';
    
	//	DEAL.II's default quad numbering: 
	// 2 * * 3
	// *     *
	// *     *
	// 0 * * 1

	// Split into 2 triangles, [0,2,3] and [0,3,1].
	
	// Write .pos file for each triangle using Tensor Triangle (TT) structure.
    // TT(x1,y1,z1,x2,y2,z2,x3,y3,z3){M1, M2, M3};
    // where Mi = m11, m12, m13, m21, m22, m23, m31, m32, m33;
	std::vector<dealii::types::global_dof_index> dof_indices(fe_system.dofs_per_cell);

	for(const auto &cell : dof_handler_vertices.active_cell_iterators())
	{
		if(! cell->is_locally_owned()) {continue;}
		cell->get_dof_indices(dof_indices);
		
		for(const auto &tri :
		            std::array<std::array<int,3>,2>{{
                    {{0,2,3}},
                    {{0,3,1}} }} )
		{
			outfile<<"TT(";
			// Write vertices
			for(unsigned int i = 0; i < tri.size(); ++i)
			{
                if(i != 0) {outfile << ",";}
				
				const unsigned int idof = tri[i];
				const unsigned int idof_global = dof_indices[idof];
				// x
				const dealii::Point<dim> &pos = all_vertices[idof_global];
				if(dim >= 1){outfile << pos[0] << ",";}
				else        {outfile << 0      << ",";}
				// y
				if(dim >= 2){outfile << pos[1] << ",";}
				else        {outfile << 0      << ",";}
				// z
				if(dim >= 3){outfile << pos[2];}
				else        {outfile << 0;}
			}
			outfile<<"){";
			// Write metric
			bool flag = false;
			
			for(unsigned int i = 0; i < tri.size(); ++i)
			{
				const unsigned int idof = tri[i];
				const unsigned int idof_global = dof_indices[idof];
				
				const dealii::Tensor<2,dim,real> &vertex_metric = optimal_metric_at_vertices[idof_global];

				for(unsigned int idim =0; idim < 3; ++idim)
				{
					for(unsigned int jdim =0; jdim < 3; ++jdim)
					{
						if(flag) {outfile<<",";}
						else	 {flag = true;}

						if( (idim < dim) && (jdim < dim) )
						{ 
							outfile<<vertex_metric[idim][jdim]; 
						}
						else
						{
							// output 1 along diagonal
							if(idim == jdim) {outfile<<1.0;}
							else			 {outfile<<0;}
						}
					}
				}
			}

			outfile<<"};"<<'\n';

		} //loop on tri : std::array<std::array>> ends
	} // cell loop ends

	outfile<<"};"<<'\n';
	outfile.close();
}

template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: write_geo_file()
{
	// Based on gmsh/tutorials/t17.geo
	std::ofstream outfile(filename_geo);
	// Header
	outfile<<" // GEO file "<<'\n'<<'\n';
	const std::string quotes = "\"";

	outfile<<"SetFactory("<<quotes<<"OpenCASCADE"<<quotes<<");"<<'\n'<<'\n';

	// Geo file of rectangle
	GridRefinement::GmshOut<dim, real>::write_geo_hyper_cube(0.0, 1.0, outfile, true);

	// Merge .pos view
	outfile<<"Merge "<<quotes<<filename_pos<<quotes<<";"<<'\n';

	// Apply the view as the current background mesh
	outfile<<"Background Mesh View[0];"<<'\n';
	
	// Use BAMG (Algorithm 7) to generate mesh.
	outfile<<"Mesh.SmoothRatio = 0;"<<'\n'; // No smoothing for now.
	outfile<<"Mesh.AnisoMax = 1e30;"<<'\n';
	outfile<<"Mesh.Algorithm = 7;"<<'\n';

	// Recombine triangles into quads.
	outfile<<"Mesh.RecombinationAlgorithm = 2;"<<'\n';
	outfile<<"Mesh.RecombineAll = 1;"<<'\n';
	outfile.close();
}

// Instantiations
template class MetricToMeshGenerator <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class MetricToMeshGenerator <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class MetricToMeshGenerator <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // PHiLiP namespace
