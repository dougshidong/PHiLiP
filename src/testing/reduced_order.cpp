#include <fstream>
#include <iostream>
#include <filesystem>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include "reduced_order.h"
#include "parameters/all_parameters.h"
#include <deal.II/lac/full_matrix.h>
#include "parameters/parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver.h"


namespace PHiLiP {
    namespace Tests {

        template <int dim, int nstate>
        ReducedOrder<dim, nstate>::ReducedOrder(const PHiLiP::Parameters::AllParameters *const parameters_input)
                : TestsBase::TestsBase(parameters_input)
        {}

        template <int dim, int nstate>
        int ReducedOrder<dim, nstate>::run_test() const
        {
            //Testing for how to extract vectors from file and make POD basis. Will likely end up being a class of its own.
            const Parameters::AllParameters param = *(TestsBase::all_parameters);

            std::vector<dealii::FullMatrix<double>> snapshotMatrixContainer;
            std::string path = "snapshot_generation"; //Search this directory for solutions_table.txt files
            for (const auto & entry : std::filesystem::recursive_directory_iterator(path)){ //Recursive seach
                if(entry.path().filename() == "solutions_table.txt"){
                    pcout << "Processing " << entry.path() << std::endl;
                    std::ifstream myfile(entry.path());
                    if(!myfile)
                    {
                        pcout<<"Error opening file. Please ensure that solutions_table.txt is located in the current directory."<< std::endl;
                        return -1;
                    }
                    std::string line;
                    int rows = 0;
                    int cols = 0;
                    //First loop set to count rows and columns
                    while(std::getline(myfile, line)){ //for each line
                        std::istringstream stream(line);
                        std::string field;
                        cols = 0;
                        while (getline(stream, field,' ')){ //parse data values on each line
                            if (field.empty()){ //due to whitespace
                                continue;
                            }else{
                                cols++;
                            }
                        }
                        rows++;
                    }

                    dealii::FullMatrix<double> solutions_matrix(rows-1, cols); //Subtract 1 from row because of header row
                    rows = 0;
                    myfile.clear();
                    myfile.seekg(0); //Bring back to beginning of file
                    //Second loop set to build solutions matrix
                    while(std::getline(myfile, line)){ //for each line
                        std::istringstream stream(line);
                        std::string field;
                        cols = 0;
                        if(rows != 0){
                            while (getline(stream, field,' ')) { //parse data values on each line
                                if (field.empty()) {
                                    continue;
                                } else {
                                    solutions_matrix.set(rows - 1, cols, std::stod(field));
                                    cols++;
                                }
                            }
                        }
                        rows++;
                    }
                    myfile.close();
                    snapshotMatrixContainer.push_back(solutions_matrix);
                }
            }

            int numMat = snapshotMatrixContainer.size();
            int totalCols = 0;
            std::vector<int> j_offset;
            for(int i = 0; i < numMat; i++){
                totalCols = totalCols + snapshotMatrixContainer[i].n_cols();
                if (i == 0){
                    j_offset.push_back(0);
                }else{
                    j_offset.push_back(j_offset[i-1] + snapshotMatrixContainer[i-1].n_cols());
                }
            }

            dealii::LAPACKFullMatrix<double> snapshot_matrix(snapshotMatrixContainer[0].n_rows(), totalCols);

            for(int i = 0; i < numMat; i++){
                dealii::FullMatrix<double> snapshot_submatrix = snapshotMatrixContainer[i];
                snapshot_matrix.fill(snapshot_submatrix, 0, j_offset[i], 0, 0);
            }

            pcout << "Snapshot matrix generated." << std::endl;

            std::ofstream out_file("snapshot_matrix.txt");
            snapshot_matrix.print_formatted(out_file, 4);

            return 0; //need to change
        }
#if PHILIP_DIM==1
        template class ReducedOrder<PHILIP_DIM,PHILIP_DIM>;
#endif
    } // Tests namespace
} // PHiLiP namespace
