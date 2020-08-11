#include <iostream>
#include <fstream>

#include <deal.II/base/utilities.h>

#include "mesh/free_form_deformation.h"


int main (int argc, char * argv [])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const unsigned int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank == 0);

    const double TOL = 1e-12;
    bool fail_bool = false;

    // Create a FFD box with origin {1,2,3} with 3, 4, and 5 control points
    // such that each interval is of length 1.
    const int dim = PHILIP_DIM;
    const dealii::Point<3>            _origin = {1,2,3};
    const std::array<double,3>        _rectangle_lengths = {{2,3,4}};
    const std::array<unsigned int,3>  _ndim_control_pts = {{3,4,5}};

    // We will not move the first section.
    dealii::Point<3> _undisplaced_point_location;
    // We will definitely the last "upper" section.
    dealii::Point<3> _displaced_point_location;
    for (int d=0; d<3; ++d) {
        _undisplaced_point_location[d] = _origin[d] - 0.1;
        _displaced_point_location[d] = _origin[d] + _rectangle_lengths[d] - 0.1;
    }

    dealii::Point<dim>                 origin;
    std::array<double,dim>             rectangle_lengths;
    std::array<unsigned int,dim>       ndim_control_pts;

    dealii::Point<dim> undisplaced_point_location;
    dealii::Point<dim> displaced_point_location;
    for (int d=0; d<dim; ++d) {
        origin[d] = _origin[d];
        rectangle_lengths[d] = _rectangle_lengths[d];
        ndim_control_pts[d] = _ndim_control_pts[d];

        undisplaced_point_location[d] = _undisplaced_point_location[d];
        displaced_point_location[d] = _displaced_point_location[d];
    }
    PHiLiP::FreeFormDeformation<dim> ffd (origin, rectangle_lengths, ndim_control_pts);

    // Point on a control point should move by the exact amount as the control point.
    const dealii::Point<dim> &control_point_to_follow = ffd.control_pts[ffd.n_control_pts-1];
    dealii::Point<dim> point_on_control = control_point_to_follow;

    dealii::Point<dim> new_point;


    pcout << " Testing unperturbed control points... " << std::endl;

    new_point = ffd.new_point_location(undisplaced_point_location);
    if ((new_point - undisplaced_point_location).norm() > TOL) {
        pcout << " Before moving control points, point " << undisplaced_point_location
              << " should have been undisplaced, but is now located at " << new_point
              << std::endl;
        fail_bool = true;
    }

    new_point = ffd.new_point_location(displaced_point_location);
    if ((new_point - displaced_point_location).norm() > TOL) {
        pcout << " Before moving control points, point " << displaced_point_location
              << " should have been undisplaced, but is now located at " << new_point
              << std::endl;
        fail_bool = true;
    }

    new_point = ffd.new_point_location(point_on_control);
    if ((new_point - point_on_control).norm() > TOL) {
        pcout << " Before moving control points, point " << point_on_control
              << " should have been undisplaced, but is now located at " << new_point
              << std::endl;
        fail_bool = true;
    }

    pcout << " Testing perturbed control points... " << std::endl;

    for (unsigned int ictl = 0; ictl < ffd.n_control_pts; ++ictl) {
        std::array<unsigned int, dim> ijk = ffd.global_to_grid(ictl);

        dealii::Tensor<1,dim,double> dx;
        for (int d=0; d<dim; ++d) {
            // Displace the uppper half the volume
            if (ijk[d] >= ndim_control_pts[d] / 2.0) {
                dx[d] = -0.5 * ijk[d]/(ndim_control_pts[d]-1);
            }
        }
        pcout << " Control point ijk: ";
        for (int d=0; d<dim; ++d) { pcout << ijk[d] << " "; }
        pcout << " located at: " << ffd.control_pts[ictl] << " moved to ";
        ffd.move_ctl_dx (ictl, dx);
        pcout << ffd.control_pts[ictl] << std::endl;
    }

    new_point = ffd.new_point_location(undisplaced_point_location);
    if ((new_point - undisplaced_point_location).norm() > TOL) {
        pcout << " Point in unperturbed section: " << undisplaced_point_location
              << " should have been undisplaced, but is now located at " << new_point
              << std::endl;
        fail_bool = true;
    }

    new_point = ffd.new_point_location(displaced_point_location);
    if ((new_point - displaced_point_location).norm() < TOL) {
        pcout << " Point in perturbed section " << displaced_point_location
              << " should have been displaced, but is still located at " << new_point
              << std::endl;
        fail_bool = true;
    }

    new_point = ffd.new_point_location(point_on_control);
    if ((new_point - control_point_to_follow).norm() > TOL) {
        pcout << " Before moving control points, point " << point_on_control
              << " should have been undisplaced to the same location as control point: " << control_point_to_follow
              << ", but is now located at " << new_point
              << std::endl;
        fail_bool = true;
    }

    const int nx = 100;
    const int ny = (dim>=3) ? nx : 1;

    const int npts = nx*ny;
    std::vector<dealii::Point<dim>> grid_points(npts);

    const double pi = atan(1.0)*4.0;
    const double frequency = 2.0;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const int ij = j * nx + i;
            //grid_points[ij][0] = origin[d] + i/(nx-1.0) * rectangle_lengths[d];
            grid_points[ij][0] = i/(nx-1.0);
            if (dim == 2) {
                grid_points[ij][1] = std::sin(grid_points[ij][0] * frequency*2*pi) / 2.0 + 0.5;
            } else if (dim == 3) {
                grid_points[ij][1] = j/(ny-1.0);
                grid_points[ij][2] = std::sin(grid_points[ij][0] * frequency*2*pi) * std::sin(grid_points[ij][1] * frequency*2*pi) / 2.0 + 0.5;
            }
            for (int d=0; d<dim; ++d) {
                grid_points[ij][d] *= rectangle_lengths[d];
                grid_points[ij][d] += origin[d];
            }
        }
    }


    if ( mpi_rank == 0 ) {
        std::ofstream initial_sine_points_file, deformed_sine_points_file;
        initial_sine_points_file.open ("initial_sine_points.dat");
        deformed_sine_points_file.open ("deformed_sine_points.dat");

        std::vector<dealii::Point<dim>> sine_points(npts);
        for (int i = 0; i<npts; ++i) {
            new_point = ffd.new_point_location(grid_points[i]);
            initial_sine_points_file << grid_points[i] << std::endl;
            deformed_sine_points_file  << new_point << std::endl;
        }

        initial_sine_points_file.close();
        deformed_sine_points_file.close();

        std::cout << "To plot a sine wave being deformated, use the following command in the terminal" << std::endl;
        if (dim == 2) std::cout <<  " gnuplot -e \"set terminal jpeg; plot 'tests/unit_tests/grid/initial_sine_points.dat' with linespoints, 'tests/unit_tests/grid/deformed_sine_points.dat' with linespoints; set xrange [1:3]; set yrange[2:5] \" > out.jpeg "  << std::endl;

        if (dim == 3) std::cout << " gnuplot -e \"set terminal jpeg; splot 'tests/unit_tests/grid/initial_sine_points.dat' using 1:2:3, 'tests/unit_tests/grid/deformed_sine_points.dat' using 1:2:3; set xrange [1:3]; set yrange[2:5]; set zrange[3:7] > out.jpeg " << std::endl;
    }


    if (fail_bool) {
        pcout << "Test failed." << std::endl;
    } else {
        pcout << "Test successful." << std::endl;
    }
    return fail_bool;
}
