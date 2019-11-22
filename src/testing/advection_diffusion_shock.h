#ifndef __ADVECTION_DIFFUSION_SHOCK_H__
#define __ADVECTION_DIFFUSION_SHOCK_H__

#include <deal.II/base/tensor.h>

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "physics/convection_diffusion.h"
#include "physics/manufactured_solution.h"
#include "parameters/all_parameters.h"
#include "functional/functional.h"

namespace PHiLiP{
namespace Tests{

// test with manufactured solution from a tensor product of arc sines

// new manufactured solution
template <int dim, typename real>
class ManufacturedSolutionShocked : public ManufacturedSolutionFunction<dim,real>
{
// We want the Point to be templated on the type,
// however, dealii does not template that part of the Function.
// Therefore, we end up overloading the functions and need to "import"
// those non-overloaded functions to avoid the warning -Woverloaded-virtual
// See: https://stackoverflow.com/questions/18515183/c-overloaded-virtual-function-warning-by-clang
protected:
    using dealii::Function<dim,real>::value;
    using dealii::Function<dim,real>::gradient;
    using dealii::Function<dim,real>::hessian;
public:
    // constructor
    ManufacturedSolutionShocked(const unsigned int /*nstate*/)
    {
        n_shocks.resize(dim);
        S_j.resize(dim);
        x_j.resize(dim);

        for(unsigned int i = 0; i<dim; ++i){
            n_shocks[i] = 2;

            S_j[i].resize(n_shocks[i]);
            x_j[i].resize(n_shocks[i]);

            S_j[i][0] =  50;
            S_j[i][1] = -50;

            x_j[i][0] = -0.3712351314;
            x_j[i][1] =  0.4523462124;
        }
    }

    // overriding the function value
    real value(const dealii::Point<dim,real> &pos, const unsigned int istate = 0) const override;

    // overriding the gradient
    dealii::Tensor<1,dim,real>gradient(const dealii::Point<dim,real> &pos, const unsigned int istate = 0) const override;

    // overriding the hessian
    dealii::SymmetricTensor<2,dim,real> hessian(const dealii::Point<dim,real> &pos, const unsigned int istate = 0) const override;

private:
    std::vector<unsigned int> n_shocks; // number of shocks
    std::vector<std::vector<real>> S_j; // shock strengths
    std::vector<std::vector<real>> x_j; // shock positions
};

template <int dim, int nstate>
class AdvectionDiffusionShock : public TestsBase
{
public: 
    // deleting the default constructor
    AdvectionDiffusionShock() = delete;

    // Constructor to call the TestsBase constructor to set parameters = parameters_input
    AdvectionDiffusionShock(const Parameters::AllParameters *const parameters_input);

    // destructor 
    ~AdvectionDiffusionShock(){};

    // perform test 
    int run_test() const;
};


} // namespace Tests
} // namespace PHiLiP

#endif // __ADVECTION_DIFFUSION_SHOCK_H__