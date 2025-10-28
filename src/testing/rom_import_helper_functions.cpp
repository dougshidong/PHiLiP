#include "rom_import_helper_functions.h"
#include "reduced_order/pod_basis_offline.h"
#include "parameters/all_parameters.h"
#include <eigen/Eigen/Dense>
#include <iostream>
#include <filesystem>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

namespace PHiLiP {
namespace Tests {

bool getSnapshotParamsFromFile(Eigen::MatrixXd& snapshot_parameters, std::string path) {
    bool file_found = false;

    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end()); //Sort files so that the order is the same as for the sensitivity basis

    for (const auto & entry : files_in_directory){
        if(std::string(entry.filename()).std::string::find("snapshot_table") != std::string::npos){
            file_found = true;
            std::ifstream myfile(entry);
            if(!myfile)
            {
                std::abort();
            }
            std::string line;
            int rows = 0;
            int cols = 0;
            //First loop set to count rows and columns
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                cols = 0;
                bool any_entry = false;
                while (getline(stream, field,' ')){ //parse data values on each line
                    if (field.empty()){ //due to whitespace
                        continue;
                    } try{
                        std::stod(field);
                        cols++;
                        any_entry = true;
                    } catch (...){
                        continue;
                    } 
                }
                if (any_entry){
                    rows++;
                }
                
            }

            int row;
            if (snapshot_parameters.rows() == 0){
                row = 0;
               snapshot_parameters.conservativeResize(rows, snapshot_parameters.cols()+cols);
                
            }
            else{
                row = snapshot_parameters.rows(); 
                snapshot_parameters.conservativeResize(snapshot_parameters.rows() + rows, snapshot_parameters.cols());
            }
            
            myfile.clear();
            myfile.seekg(0); //Bring back to beginning of file
            //Second loop set to build solutions matrix
            while(std::getline(myfile, line)){ //for each line
                std::istringstream stream(line);
                std::string field;
                int col = 0;
                bool any_entry = false;
                while (getline(stream, field,' ')) { //parse data values on each line
                    if (field.empty()) {
                        continue;
                    }
                    try{
                        double num_string = std::stod(field);
                        snapshot_parameters(row, col) = num_string;
                        col++;
                        any_entry = true;
                    } catch (...){
                        continue;
                    }
                }
                if (any_entry){
                    row++;
                }
            }
            myfile.close();
        }
    }
    return file_found;
}

void getROMPoints(Eigen::MatrixXd& rom_points, const Parameters::AllParameters *const all_parameters) {
    const double pi = atan(1.0) * 4.0;
    if(all_parameters->reduced_order_param.parameter_names.size() == 1){
        rom_points.conservativeResize(20, 1);
        RowVectorXd parameter1_range;
        parameter1_range.resize(2);
        parameter1_range << all_parameters->reduced_order_param.parameter_min_values[0], all_parameters->reduced_order_param.parameter_max_values[0];
        if(all_parameters->reduced_order_param.parameter_names[0] == "alpha"){
            parameter1_range *= pi/180; // convert to radians
        }

        double step_1 = (parameter1_range[1] - parameter1_range[0]) / (20 - 1);

        int row = 0;
        for (int i = 0; i < 20; i++){
            rom_points(row, 0) =  parameter1_range[0] + (step_1 * i);
            row ++;
        }
    }
    else{
        rom_points.conservativeResize(400, 2);
        RowVectorXd parameter1_range;
        parameter1_range.resize(2);
        parameter1_range << all_parameters->reduced_order_param.parameter_min_values[0], all_parameters->reduced_order_param.parameter_max_values[0];

        RowVectorXd parameter2_range;
        parameter2_range.resize(2);
        parameter2_range << all_parameters->reduced_order_param.parameter_min_values[1], all_parameters->reduced_order_param.parameter_max_values[1];
        if(all_parameters->reduced_order_param.parameter_names[1] == "alpha"){
            parameter2_range *= pi/180; //convert to radians
        }
        double step_1 = (parameter1_range[1] - parameter1_range[0]) / (20 - 1);
        double step_2 = (parameter2_range[1] - parameter2_range[0]) / (20 - 1);

        int row = 0;
        for (int i = 0; i < 20; i++){
            for(int j = 0; j < 20; j++){
                rom_points(row, 0) =  parameter1_range[0] + (step_1 * i);
                rom_points(row, 1) =  parameter2_range[0] + (step_2 * j);
                row ++;
            }
        }
    }
}

} // End of Tests namespace
} // End of PHiLiP namespace