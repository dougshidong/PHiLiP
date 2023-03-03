#ifndef __ANISOTROPIC_MESH_ADAPTATION__
#define __ANISOTROPIC_MESH_ADAPTATION__

#include "dg/dg.h"
#include <deal.II/base/symmetric_tensor.h>

namespace PHiLiP {

/** Performs anisotropic mesh adaptation with an optimal metric for P1 solution approximation.
 *  Implements the optimal metric field derived from continuous optimization framework. 
 *  See papers from INRIA for further details: 
 *  Feature based : Loseille, A. and Alauzet, F. "Continuous mesh framework part I: well-posed continuous interpolation error.", 2011.
 *  Goal oriented: Loseille, A., Dervieux, A., and Alauzet, F. "Fully anisotropic goal-oriented mesh adaptation for 3d steady Euler equations.", 2010.
 * @note The goal oriented approach is currently implemented for convection dominated flows.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class AnisotropicMeshAdaptation {
	
	using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.

public:
	/// Constructor
	AnisotropicMeshAdaptation(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, const bool _use_goal_oriented_approach = false);

	/// Destructor
	~AnisotropicMeshAdaptation(){};

	/// Returns positive tensor from an input tensor, by taking absolute eigenvalues.
	dealii::SymmetricTensor<2, dim, real> get_positive_definite_tensor(const dealii::Tensor<2, dim, real> &input_tensor);

	/// Computes optimal metric depending on goal oriented or feature based approach. 
	void compute_optimal_metric();


protected:
	/// Computes hessian using the input coefficients, which can be a solution sensor or (for goal oriented approach) convective flux.
	/** This function is called by compute_optimal_metric(). 
	 */
	void compute_hessian(const VectorType & sensor_coeffs);

    /// Shared pointer to DGBase.
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

	///Flag to use goal oriented approach. It is set to false by default.
	const bool use_goal_oriented_approach;

	/// Stores optimal metric in each cell
	std::vector<dealii::Tensor<2, dim, real>> cellwise_optimal_metric;
	
	/// Stores hessian in each cell
	std::vector<dealii::Tensor<2, dim, real>> cellwise_hessian;
};

} // PHiLiP namepsace

#endif
