#ifndef __PHILIP_GMSH_READER_H__
#define __PHILIP_GMSH_READER_H__

// Gmsh - Copyright (C) 1997-2020 C. Geuzaine, J.-F. Remacle
//
// See the LICENSE.txt file for license information. Please report all
// issues on https://gitlab.onelab.info/gmsh/gmsh/issues.

#ifndef GMSH_DEFINES_H
#define GMSH_DEFINES_H

// Element types in .msh file format (numbers should not be changed)
//
// POINT
#define MSH_PNT      15
// LINE
#define MSH_LIN_2    1
#define MSH_LIN_3    8
#define MSH_LIN_4    26
#define MSH_LIN_5    27
#define MSH_LIN_6    28
#define MSH_LIN_7    62
#define MSH_LIN_8    63
#define MSH_LIN_9    64
#define MSH_LIN_10   65
#define MSH_LIN_11   66
// QUAD
#define MSH_QUA_4    3
#define MSH_QUA_9    10
#define MSH_QUA_16   36
#define MSH_QUA_25   37
#define MSH_QUA_36   38
#define MSH_QUA_49   47
#define MSH_QUA_64   48
#define MSH_QUA_81   49
#define MSH_QUA_100  50
#define MSH_QUA_121  51
// HEXES COMPLETE (3->9)
#define MSH_HEX_8    5
#define MSH_HEX_27   12
#define MSH_HEX_64   92
#define MSH_HEX_125  93
#define MSH_HEX_216  94
#define MSH_HEX_343  95
#define MSH_HEX_512  96
#define MSH_HEX_729  97
#define MSH_HEX_1000 98

#endif

#include <deal.II/base/exceptions.h>
#include "high_order_grid.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
    /// Reads Gmsh grid.
    /** Can request to convert the input grid's order to the 
      * requested_grid_order, which will simply interpolate
      * the high-order nodes. Dealii's mesh smoothing can be set to none
      * while using goal oriented mesh adaptation.
      */
    template <int dim, int spacedim>
    std::shared_ptr< HighOrderGrid<dim, double> >
    read_gmsh(std::string filename, const bool periodic_x, 
              const bool periodic_y, const bool periodic_z, 
              const int x_periodic_1, const int x_periodic_2, 
              const int y_periodic_1, const int y_periodic_2, 
              const int z_periodic_1, const int z_periodic_2, 
              const bool mesh_reader_verbose_output,
              int requested_grid_order,
              const bool use_mesh_smoothing);

    /** Reads Gmsh grid from file with all other arguments as default.
     *  See function definition for what the defaults are set to.
    */
    template <int dim, int spacedim>
    std::shared_ptr< HighOrderGrid<dim, double> >
    read_gmsh(std::string filename);

    /** Reads Gmsh grid from file at a given requested_grid_order and use_mesh_smoothing input;
     *  with all other arguments as default.
     *  See function definition for what the defaults are set to.
    */
    template <int dim, int spacedim>
    std::shared_ptr< HighOrderGrid<dim, double> >
    read_gmsh(std::string filename, int requested_grid_order, const bool use_mesh_smoothing);

    /** Reads Gmsh grid from file that has periodic boundaries;
     *  with all other arguments as default.
     *  See function definition for what the defaults are set to.
    */
    template <int dim, int spacedim>
    std::shared_ptr< HighOrderGrid<dim, double> >
    read_gmsh(std::string filename, const bool periodic_x, 
              const bool periodic_y, const bool periodic_z, 
              const int x_periodic_1, const int x_periodic_2, 
              const int y_periodic_1, const int y_periodic_2, 
              const int z_periodic_1, const int z_periodic_2, 
              const bool mesh_reader_verbose_output);

    
} // namespace PHiLiP
#endif

