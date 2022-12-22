/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2001 - 2014 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, Ralf Hartmann, University of Heidelberg, 2001
 * Modified from step-10 by Juan Carlos Araújo, 5/apr/2017
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include <iostream>
#include <fstream>
#include <cmath>

#include <deal.II/numerics/data_out.h>
#define PI 3.14159265358979323846

namespace Step10
{
    using namespace dealii;
      
      // Description of the upper face of the square
    class UpperManifold: public ChartManifold<2,2,2> {
    public:
        virtual Point<2> pull_back(const Point<2> &space_point) const; ///< See dealii::Manifold::pull_back().
        virtual Point<2> push_forward(const Point<2> &chart_point) const; ///< See dealii::Manifold::pull_forward().
        virtual std::unique_ptr<Manifold<2,2> > clone() const; ///< See dealii::Manifold::clone().
    };
  
    Point<2> UpperManifold::pull_back(const Point<2> &space_point) const {
        double x_phys = space_point[0];
        double y_phys = space_point[1];
        double x_ref = x_phys;//(x_phys+1.5)/3.0;
        double y_ref = 0.5;

        for (int i=0; i<20; i++) {
            const double function = 0.8*y_ref + exp(-30*y_ref*y_ref)*0.0625*exp(-25*x_phys*x_phys) - y_phys;
            const double derivative = 0.8 + -30*y_ref*exp(-30*y_ref*y_ref)*0.0625*exp(-25*x_phys*x_phys);
            y_ref = y_ref - function/derivative;
        }

        Point<2> p(x_ref, y_ref);
        return p;
    }
  
    Point<2> UpperManifold::push_forward(const Point<2> &chart_point) const {
        double x_ref = chart_point[0];
        double y_ref = chart_point[1];
        // return Point<2> (x_ref, -2*x_ref*x_ref + 2*x_ref + 1);   // Parabole 
        double x_phys = x_ref;//-1.5+x_ref*3.0;
        double y_phys = 0.8*y_ref + exp(-30*y_ref*y_ref)*0.0625*exp(-25*x_phys*x_phys);
        //return Point<2> ( -1.5+x_ref*3.0, 0.8*y_ref + exp(-10*y_ref*y_ref)*0.0625*exp(-25*x_ref*x_ref) ); // Trigonometric
        //return Point<2> ( x_phys, y_phys ); // Trigonometric
        return Point<2> ( x_phys, y_phys); // Trigonometric
    }
  
    std::unique_ptr<Manifold<2,2> > UpperManifold::clone() const
    {
        return std::make_unique<UpperManifold>();
    }
  
    template <int dim>
    void compute_area ()
    {
        const double exact_bump_integral = 0.02215567313631895;
              
        // The case p=1, is trivial: no bending!
        for (unsigned int degree=1; degree<6; ++degree) {
            std::cout << "Degree = " << degree << std::endl;

            Triangulation<dim> triangulation;

            const bool colorize = false;
            Point<2> p1(-1.5,0.0), p2(1.5,0.8);
            GridGenerator::hyper_rectangle (triangulation, p1, p2, colorize);
            
            unsigned int manifold_id=0; // top face, see GridGenerator::hyper_rectangle, colorize=true
            static const UpperManifold manifold;
            triangulation.set_all_manifold_ids(0);
            triangulation.set_manifold ( manifold_id, manifold );
            
            const MappingQ<dim> mapping (degree+1);
            //const MappingManifold<dim> mapping ();
            const QGauss<dim> quadrature(degree+1);
            
            //const FE_DGQ<dim> dummy_fe (QGauss<1>(degree + 1));
            const FE_DGQ<dim> dummy_fe (degree + 1);
            DoFHandler<dim> dof_handler (triangulation);
                
            FEValues<dim> fe_values (mapping, dummy_fe, quadrature, update_JxW_values);

            ConvergenceTable table;
            
            for (unsigned int refinement=0; refinement<6; ++refinement, triangulation.refine_global (1))
            {
                table.add_value("cells", triangulation.n_active_cells());

                dof_handler.distribute_dofs (dummy_fe);

                DataOut<dim> data_out;
                data_out.attach_dof_handler (dof_handler);

                Vector<double> dummy ( dof_handler.n_dofs() );
                std::string name = "grid";
                data_out.add_data_vector (dummy, name);

                std::ostringstream ss;    
                ss <<"grid_p" << degree << "_ref" << refinement << ".vtk";
                std::ofstream output (ss.str().c_str());

                data_out.build_patches (mapping, degree+1, DataOut<dim>::curved_inner_cells);

                data_out.write_vtk (output);

                long double area = 0;

                typename DoFHandler<dim>::active_cell_iterator
                    cell = dof_handler.begin_active(),
                    endc = dof_handler.end();
                for (; cell!=endc; ++cell) {

                    fe_values.reinit (cell);
                    for (unsigned int i=0; i<fe_values.n_quadrature_points; ++i) {
                        area += fe_values.JxW (i);
                    }
                };

                double new_area = 3.0*0.8-area; // not working

                table.add_value("eval.pi", static_cast<double> (new_area));
                table.add_value("error",   static_cast<double> (std::fabs(new_area-exact_bump_integral)));

                //if (refinement == 0) {

                //}
            };
                
            table.omit_column_from_convergence_rate_evaluation("cells");
            table.omit_column_from_convergence_rate_evaluation("eval.pi");
            table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
            
            table.set_precision("eval.pi", 16);
            table.set_scientific("error", true);
            
            table.write_text(std::cout);

            std::cout << std::endl;
          };
    }
}
  
  int main ()
  {
    try
      {
        std::cout.precision (16);
        Step10::compute_area<2> ();
      }
    catch (std::exception &exc)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
  
        return 1;
      }
    catch (...)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
      }
  
    return 0;
}


