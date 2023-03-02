#include "anisotropic_mesh_adaptation.h"

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
AnisotropicMeshAdaptation<dim, nstate, real, MeshType> :: AnisotropicMeshAdaptation(
	std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, 
	const bool _use_goal_oriented_approach)
	: dg(dg_input)
	, use_goal_oriented_approach(_use_goal_oriented_approach)
	{}
	
} // PHiLiP namespace
