#ifndef __DEALII_SOLVER_ROL_VECTOR_H__
#define __DEALII_SOLVER_ROL_VECTOR_H__

#include "ROL_Types.hpp"
#include "ROL_Vector_SimOpt.hpp"


#include "optimization/rol_to_dealii_vector.hpp"

template<typename Real = double>
class dealiiSolverVectorWrappingROL
{
private:
    /// Pointer to ROL::Vector<Real> where data is actually stored.
    ROL::Ptr<ROL::Vector<Real>> rol_vector_ptr;


    void print(const ROL::Vector<Real> &rol_vector) const
    {
        const ROL::Vector_SimOpt<Real> *vec_split12 = dynamic_cast<const ROL::Vector_SimOpt<Real> *>(&rol_vector);
        if (vec_split12 == NULL) {
            PHiLiP::ROL_vector_to_dealii_vector_reference(rol_vector).print(std::cout,14);
        } else {
            const auto vec_1 = vec_split12->get_1();
            print(*vec_1);
            const auto vec_2 = vec_split12->get_2();
            print(*vec_2);
        }
    }
public:
    /// Value type of the entries.
    using value_type = Real;

    /// Constructor.
    /** Must call reinit on the vector to have something valid.
     */
    dealiiSolverVectorWrappingROL() {};

    /// Constructor where data is given.
    dealiiSolverVectorWrappingROL(ROL::Ptr<ROL::Vector<Real>> input_vector)
    : rol_vector_ptr(input_vector)
    {};

    /// Accessor.
    ROL::Ptr<ROL::Vector<Real>> getVector()
    {
        return rol_vector_ptr;
    }

    /// Const accessor.
    ROL::Ptr<const ROL::Vector<Real>> getVector() const
    {
        return rol_vector_ptr;
    }
    
    /// Resize the current object to have the same size and layout as
    /// the model_vector argument provided. The second argument
    /// indicates whether to clear the current object after resizing.
    void reinit (const dealiiSolverVectorWrappingROL &model_vector,
                 const bool leave_elements_uninitialized = false)
    {
        rol_vector_ptr = model_vector.getVector()->clone();
        if (!leave_elements_uninitialized) {
            (*this) *= 0.0;
        }
    }
    /// Inner product between the current object and the argument.
    Real operator * (const dealiiSolverVectorWrappingROL &v) const
    {
        return rol_vector_ptr->dot( *(v.getVector()) );
    }

    /// Assignment of a scalar
    dealiiSolverVectorWrappingROL & operator=(const Real a)
    {
        rol_vector_ptr->setScalar(a);
        return *this;
    }

    /// Copy assignment
    dealiiSolverVectorWrappingROL & operator=(const dealiiSolverVectorWrappingROL &x)
    {
        rol_vector_ptr = (x.getVector())->clone();
        return *this;
    }
    
    /// Scale the elements of the current object by a fixed value.
    dealiiSolverVectorWrappingROL & operator*=(const Real a)
    {
        rol_vector_ptr->scale(a);
        return *this;
    }

    /// Addition of vectors
    void add (const dealiiSolverVectorWrappingROL &x)
    {
        this->add(1.0,x);
    }

    /// Scaled addition of vectors
    void add (const Real  a,
              const dealiiSolverVectorWrappingROL &x)
    {
        rol_vector_ptr->axpy(a, *(x.getVector()) );
    }

    /// Scaled addition of vectors
    void sadd (const Real  a,
               const Real  b,
               const dealiiSolverVectorWrappingROL &x)
    {
        (*this) *= a;
        this->add(b, x);
    }

    /// Scaled assignment of a vector
    void equ (const Real  a,
              const dealiiSolverVectorWrappingROL &x)
    {
        rol_vector_ptr->set( *(x.getVector()) );
        (*this) *= a;
    }

    /// Combined scaled addition of vector x into the current object and
    /// subsequent inner product of the current object with v.
    Real add_and_dot (const Real  a,
                        const dealiiSolverVectorWrappingROL &x,
                        const dealiiSolverVectorWrappingROL &v)
    {
        this->add(a, x);
        return (*this) * v;
    }

    /// Return the l2 norm of the vector.
    Real l2_norm () const
    {
        return std::sqrt( (*this) * (*this) );
    }

    /// Returns a vector of the same size with zero entries except for the ith entry being one.
    Teuchos::RCP<dealiiSolverVectorWrappingROL> basis(int i) const
    {
        ROL::Ptr<ROL::Vector<Real>> rol_basis = rol_vector_ptr->basis(i);

        Teuchos::RCP<dealiiSolverVectorWrappingROL> e = Teuchos::rcp(new dealiiSolverVectorWrappingROL(rol_basis));

        return e;
    }

    /// Obtain vector size.
    int size() const
    {
        return rol_vector_ptr->dimension();
    }

    /// Print the underlying deal.II Vector.
    void print() const
    {
        print(*rol_vector_ptr);
        //     const auto &vec_split12 = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*rol_vector_ptr);
        //     const auto des = vec_split12.get_1();
        //     const auto &des_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*des);

        //     const auto sim_des = des_split.get_1();
        //     const auto ctl_des = des_split.get_2();
        //     const auto con = vec_split12.get_2();

        //     PHiLiP::ROL_vector_to_dealii_vector_reference(*sim_des).print(std::cout,14);
        //     PHiLiP::ROL_vector_to_dealii_vector_reference(*ctl_des).print(std::cout,14);
        //     PHiLiP::ROL_vector_to_dealii_vector_reference(*con).print(std::cout,14);
    }

    /// Access this ith value of the vector.
    /** Can not modify the value.
     */
    Real operator [](int i) const
    {
        const auto &vec_split12 = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*rol_vector_ptr);
        const auto des = vec_split12.get_1();
        const auto &des_split = dynamic_cast<const ROL::Vector_SimOpt<Real>&>(*des);

        const auto sim_des = des_split.get_1();
        const auto ctl_des = des_split.get_2();
        const auto con = vec_split12.get_2();

        const auto n1 = sim_des->dimension();
        const auto n2 = ctl_des->dimension() + n1;
        //const auto n3 = con->dimension() + n2;

        if (i < n1) {
            return PHiLiP::ROL_vector_to_dealii_vector_reference(*sim_des)[i];
        } else if (i < n2) {
            return PHiLiP::ROL_vector_to_dealii_vector_reference(*ctl_des)[i-n1];
        } else {
            return PHiLiP::ROL_vector_to_dealii_vector_reference(*con)[i-n2];
        }
    }

};

#endif
