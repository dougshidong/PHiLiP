 #include "parameters/all_parameters.h"
 #include <deal.II/base/tensor.h>
 #include "artificial_dissipation_factory.h"
 #include "artificial_dissipation.h"
 
 namespace PHiLiP {

 template <int dim, int nstate>
 std::shared_ptr < ArtificialDissipationBase<dim,nstate> >
 ArtificialDissipationFactory<dim,nstate> ::create_artificial_dissipation_pointer(const Parameters::AllParameters *const parameters_input)
 {

    if (parameters_input->physical_artificial_dissipation)
	{
		if constexpr(dim+2==nstate)
		{
			std::cout<<"Physical Artifical Dissipation pointer created"<<std::endl;
			return std::make_shared<PhysicalArtificialDissipation<dim,nstate>>(parameters_input);
		}
	}
	else 
	{
		dealii::Tensor<2,3,double> diffusion_tensor;
		diffusion_tensor[0][0]=1.0;	diffusion_tensor[0][1]=0.0; diffusion_tensor[0][2]=0.0; 
		diffusion_tensor[1][0]=0.0; diffusion_tensor[1][1]=1.0; diffusion_tensor[1][2]=0.0; 
		diffusion_tensor[2][0]=0.0;	diffusion_tensor[2][1]=0.0; diffusion_tensor[2][2]=1.0;
		std::cout<<"Laplacian Artifical Dissipation pointer created"<<std::endl;
		return std::make_shared<LaplacianArtificialDissipation<dim,nstate>>(diffusion_tensor);
    }

	std::cout<<"Cannot create Physical Artificial Dissipation Pointer as dim != nstate. Null pointer is being returned"<<std::endl;

	return nullptr;
 }


 template class ArtificialDissipationFactory<PHILIP_DIM,1>; 
 template class ArtificialDissipationFactory<PHILIP_DIM,2>; 
 template class ArtificialDissipationFactory<PHILIP_DIM,3>; 
 template class ArtificialDissipationFactory<PHILIP_DIM,4>; 
 template class ArtificialDissipationFactory<PHILIP_DIM,5>; 

 } // namespace PHiLiP
