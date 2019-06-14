#ifndef __MANUFACTUREDSOLUTIONFUNCTION_H__
#define __MANUFACTUREDSOLUTIONFUNCTION_H__

#include <deal.II/lac/vector.h>

//#include <Sacado.hpp>
//
//#include "physics/physics.h"
//#include "numerical_flux/numerical_flux.h"
//#include "parameters/all_parameters.h"


namespace PHiLiP {


template <int dim, typename real>
class ManufacturedSolutionFunction : public dealii::Function<dim,real>
{
public:
    /// Constructor that initializes base_values, amplitudes, frequencies.
    ManufacturedSolutionFunction (const unsigned int nstate = 1);
    ~ManufacturedSolutionFunction() {};
  
    real value (const dealii::Point<dim> &point, const unsigned int istate = 0) const;

    dealii::Tensor<1,dim,real> gradient (const dealii::Point<dim> &point, const unsigned int istate = 0) const;

    // Virtual functions inherited from dealii::Function
    //
    // virtual real value (const Point<dim> &p,
    //                               const unsigned int  component = 0) const;
  
    // virtual void vector_value (const Point<dim> &p,
    //                           Vector<real> &values) const;
  
    // virtual void value_list (const std::vector<Point<dim> > &points,
    //                         std::vector<real> &values,
    //                         const unsigned int              component = 0) const;
  
    // virtual void vector_value_list (const std::vector<Point<dim> > &points,
    //                                std::vector<Vector<real> > &values) const;
  
    // virtual void vector_values (const std::vector<Point<dim> > &points,
    //                            std::vector<std::vector<real> > &values) const;
  
    // virtual Tensor<1,dim, real> gradient (const Point<dim> &p,
    //                                                 const unsigned int  component = 0) const;
  
    // virtual void vector_gradient (const Point<dim> &p,
    //                              std::vector<Tensor<1,dim, real> > &gradients) const;
  
    // virtual void gradient_list (const std::vector<Point<dim> > &points,
    //                            std::vector<Tensor<1,dim, real> > &gradients,
    //                            const unsigned int              component = 0) const;
  
    // virtual void vector_gradients (const std::vector<Point<dim> > &points,
    //                               std::vector<std::vector<Tensor<1,dim, real> > > &gradients) const;
  
    // virtual void vector_gradient_list (const std::vector<Point<dim> > &points,
    //                                   std::vector<std::vector<Tensor<1,dim, real> > > &gradients) const;

private:
    std::vector<real> base_values;
    std::vector<real> amplitudes;
    std::vector<dealii::Tensor<1,dim,real>> frequencies;
};

}

#endif
