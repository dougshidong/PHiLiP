#ifndef __GNU_OUT_H__
#define __GNU_OUT_H__

#include <string>
#include <vector>

namespace PHiLiP {

namespace GridRefinement {

template <typename real>
class GnuFig
{
public:
    // constructor without name set
    GnuFig();

    // constructor with setting the name
    GnuFig(
        const std::string &name_input);

    // sets the file name
    void set_name(
        const std::string &name_input);

    // sets the figure title
    void set_title(
        const std::string &title_input);

    // set the xlabel
    void set_x_label(
        const std::string &xlabel_input);

    // set the ylabel
    void set_y_label(
        const std::string &ylabel_input);

    // set flag for background grid grid setting
    void set_grid(
        const bool grid_bool_input);

    // set flag for logarithmic x-axis
    void set_x_scale_log(
        const bool xlog_bool_input);

    // set flag for logarithmic y-axis
    void set_y_scale_log(
        const bool ylog_bool_input);

    // enables whether to show the legend
    void set_legend(
        const bool legend_bool_input);

    // adds x vs y data to be plotted
    void add_xy_data(
        const std::vector<real> &x_data, 
        const std::vector<real> &y_data);

    // adds x vs y data with a legend label
    void add_xy_data(
        const std::vector<real> &x_data, 
        const std::vector<real> &y_data, 
        const std::string &      label_name);

    // main function call that writes the ouput files
    void write_gnuplot();

    // executes the gnuplot file with the default path gnuplot
    void exec_gnuplot();

private:
    // write header with settings
    void write_gnuplot_header(
        std::ostream &out);

    // write body including data outputs
    void write_gnuplot_body(
        std::ostream &out);

    // write the footer
    void write_gnuplot_footer(
        std::ostream &out);

    // write the data entry i to .dat
    void write_xy_data(
        std::ostream &     out,
        const unsigned int data_id);

    // default settings
    const std::string DEFAULT_NAME   = "GnuFig";
    const std::string DEFAULT_TITLE  = "";
    const std::string DEFAULT_XLABEL = "";
    const std::string DEFAULT_YLABEL = "";
    const bool        DEFAULT_GRID   = false;
    const bool        DEFAULT_XLOG   = false;
    const bool        DEFAULT_YLOG   = false;
    const bool        DEFAULT_LEGEND = false;

    // for the legend
    const std::string DEFAULT_LABEL_PREFIX = "data_";
    

    // properties
    std::string name;
    std::string title;
    std::string xlabel;
    std::string ylabel;
    bool        grid;
    bool        xlog;
    bool        ylog;
    bool        legend;

    // data
    std::vector<std::vector<real>> x_data_vec;
    std::vector<std::vector<real>> y_data_vec;
    std::vector<std::string>       label_name_vec;
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GNU_OUT_H__
