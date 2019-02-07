#ifndef __INTEGRATOR_H__
#define __INTEGRATOR_H__

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>

template<int dim, typename real>
class IntegratorExplicit
{
public:

    IntegratorExplicit (const dealii::DoFHandler<dim>& dof_handler)
        : 
        dof_info (dof_handler) {};

    dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
    dealii::MeshWorker::DoFInfo<dim>            dof_info;
    dealii::MeshWorker::Assembler::ResidualSimple< dealii::Vector<real>>  assembler;

};
#endif
