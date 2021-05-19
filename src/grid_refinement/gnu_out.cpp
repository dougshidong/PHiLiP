#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <deal.II/base/exceptions.h>

#include "gnu_out.h"

namespace PHiLiP {

namespace GridRefinement {

template <typename real>
GnuFig<real>::GnuFig() : GnuFig<real>::GnuFig(DEFAULT_NAME){}

template <typename real>
GnuFig<real>::GnuFig(
    const std::string &name_input) : 
        name(name_input),
        title(DEFAULT_TITLE),
        xlabel(DEFAULT_XLABEL),
        ylabel(DEFAULT_YLABEL),
        grid(DEFAULT_GRID),
        xlog(DEFAULT_XLOG),
        ylog(DEFAULT_YLOG),
        legend(DEFAULT_LEGEND){}

template <typename real>
void GnuFig<real>::set_name(
    const std::string &name_input)
{
    name = name_input;
}

template <typename real>
void GnuFig<real>::set_title(
    const std::string &title_input)
{
    title = title_input;
}

template <typename real>
void GnuFig<real>::set_x_label(
    const std::string &xlabel_input)
{
    xlabel = xlabel_input;
}

template <typename real>
void GnuFig<real>::set_y_label(
    const std::string &ylabel_input)
{
    ylabel = ylabel_input;
}

template <typename real>
void GnuFig<real>::set_grid(
    const bool grid_bool_input)
{
    grid = grid_bool_input;
}

template <typename real>
void GnuFig<real>::set_x_scale_log(
    const bool xlog_bool_input)
{
    xlog = xlog_bool_input;
}

template <typename real>
void GnuFig<real>::set_y_scale_log(
    const bool ylog_bool_input)
{
    ylog = ylog_bool_input;
}

template <typename real>
void GnuFig<real>::set_legend(
    const bool legend_bool_input)
{
    legend = legend_bool_input;
}

template <typename real>
void GnuFig<real>::add_xy_data(
    const std::vector<real> &x_data, 
    const std::vector<real> &y_data)
{
    add_xy_data(
        x_data,
        y_data,
        DEFAULT_LABEL_PREFIX + std::to_string(x_data_vec.size()));
}

template <typename real>
void GnuFig<real>::add_xy_data(
    const std::vector<real> &x_data, 
    const std::vector<real> &y_data, 
    const std::string &      label_name)
{
    Assert(x_data.size() == y_data.size(), dealii::ExcInternalError());

    x_data_vec.push_back(x_data);
    y_data_vec.push_back(y_data);
    label_name_vec.push_back(label_name);
}

template <typename real>
void GnuFig<real>::write_gnuplot()
{
    std::ofstream gnu_out(name + ".gp");

    write_gnuplot_header(gnu_out);
    write_gnuplot_body(gnu_out);
    write_gnuplot_footer(gnu_out);

    gnu_out << std::flush;
}

template <typename real>
void GnuFig<real>::write_gnuplot_header(
    std::ostream &out)
{
    out << "# *********************************** " << '\n'
        << "# * GNUPLOT OUTPUT FILE GENERATED   * " << '\n'
        << "# * AUTOMATICALLY BY PHiLiP LIBRARY * " << '\n'
        << "# *********************************** " << '\n' << '\n';

    out << "set term png" << '\n' << '\n';

    out << "set output '" << name << ".png" << "'" << '\n';
    out << "set title '" << title << "'" << '\n';
    out << "set xlabel '" << xlabel << "'" << '\n';
    out << "set ylabel '" << ylabel << "'" << '\n' << '\n';

    if(grid){
        out << "set grid" << '\n';
    }else{
        out << "unset grid" << '\n';
    }

    if(xlog){
        out << "set log x" << '\n';
        // out << "set format x '%g'" << '\n';
        out << "set format x \"10^{%L}\"" << '\n';
    }

    if(ylog){
        out << "set log y" << '\n';
        // out << "set format y '%g'" << '\n';
        out << "set format y \"10^{%L}\"" << '\n';
    }

    if(legend){
        out << "set key" << '\n';
    }else{
        out << "unset key" << '\n';
    }

    out << '\n';
}

template <typename real>
void GnuFig<real>::write_gnuplot_body(
    std::ostream &out)
{
    Assert(x_data_vec.size() == y_data_vec.size(), dealii::ExcInternalError());
    Assert(x_data_vec.size() == label_name_vec.size(), dealii::ExcInternalError());

    out << "plot ";

    for(unsigned int i = 0; i < x_data_vec.size(); ++i){
        if(i > 0)
            out << ", \\" << '\n' << "     ";
        
        std::string dat_filename = name+ "_" + label_name_vec[i] + ".dat";
        std::ofstream dat_out(dat_filename);

        write_xy_data(dat_out, i);

        out << "'" << dat_filename << "' with linespoint";

        if(legend)
            out << " title \"" << label_name_vec[i] << "\"";

    }
    out << '\n' << '\n';
}

template <typename real>
void GnuFig<real>::write_gnuplot_footer(
    std::ostream &/*out*/)
{
    // out << "quit" << '\n' << '\n';
}

template <typename real>
void GnuFig<real>::write_xy_data(
    std::ostream &     out,
    const unsigned int data_id)
{
    Assert(x_data_vec[data_id].size() == y_data_vec[data_id].size(), dealii::ExcInternalError());
    for(unsigned int i = 0; i < x_data_vec[data_id].size(); ++i)
        out <<  x_data_vec[data_id][i] << '\t' << y_data_vec[data_id][i] << '\t' << '\n';
}

template <typename real>
void GnuFig<real>::exec_gnuplot()
{
#if ENABLE_GNUPLOT
    int ret =  std::system(("gnuplot \"" + name + ".gp\"").c_str());
    (void) ret;
#else
    std::cout << "Note: gnuplot not availible. Set ENABLE_GNUPLOT to automatically run \"" 
              << name << ".gp\"" << std::endl;
#endif
}

template class GnuFig <double>;

} // namespace GridRefinement

} // namespace PHiLiP