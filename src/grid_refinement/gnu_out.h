#ifndef __GNU_OUT_H__
#define __GNU_OUT_H__

#include <string>
#include <vector>

namespace PHiLiP {

namespace GridRefinement {

/// Gnuplot utility class
/** Class provides helper functions for creating  simple formatted gnuplot figures 
  * based on .gp files which can be used to track live error convergence during mesh adaptation runs.
  */
template <typename real>
class GnuFig
{
public:
    /// Constructor with no name specified
    /** Uses default name, calls specialized constructor
      * Name can be changed later using set_name below. 
      */
    GnuFig();

    /// Constructor with name specified
    /** Sets name and default formatting for gnuplot figure.
      */ 
    GnuFig(
        const std::string &name_input);

    /// Sets the file output name (without extension)
    void set_name(
        const std::string &name_input);

    /// Sets the figure title
    void set_title(
        const std::string &title_input);

    /// Sets the x-axis label
    void set_x_label(
        const std::string &xlabel_input);

    /// Sets the y-axis label
    void set_y_label(
        const std::string &ylabel_input);

    /// Set flag for enabling background grid
    void set_grid(
        const bool grid_bool_input);

    /// Set flag for logarithmic x-axis
    void set_x_scale_log(
        const bool xlog_bool_input);

    /// Set flag for logarithmic y-axis
    void set_y_scale_log(
        const bool ylog_bool_input);

    /// Sets display visibility of figure legend
    void set_legend(
        const bool legend_bool_input);

    /// Adds 2D x vs. y data to be plotted (default legend label)
    void add_xy_data(
        const std::vector<real> &x_data, 
        const std::vector<real> &y_data);

    // Adds 2D x vs. y data to be plotted (with a legend label)
    void add_xy_data(
        const std::vector<real> &x_data, 
        const std::vector<real> &y_data, 
        const std::string &      label_name);

    /// Main write function call
    /** Outputs format specification .gp file and data files .dat for each added vector set added
      */
    void write_gnuplot();

    /// Executes the gnuplot file
    /** Requires ENABLE_GNUPLOT from the CMakeLists.txt file to peform system call
      */
    void exec_gnuplot();

private:
    /// Write the figure formatting header based on settings
    void write_gnuplot_header(
        std::ostream &out);

    /// Writes the figure body including performing data outputs
    void write_gnuplot_body(
        std::ostream &out);

    /// Writes the figure footer
    void write_gnuplot_footer(
        std::ostream &out);

    /// Write the i^th data entry to .dat file
    void write_xy_data(
        std::ostream &     out,
        const unsigned int data_id);

    // default settings
    const std::string DEFAULT_NAME   = "GnuFig"; ///< Default file name (no extension)
    const std::string DEFAULT_TITLE  = "";       ///< Default figure title
    const std::string DEFAULT_XLABEL = "";       ///< default figure x-axis label
    const std::string DEFAULT_YLABEL = "";       ///< deafult figure y-axis label 
    const bool        DEFAULT_GRID   = false;    ///< Default flag for enabling grid line visibility
    const bool        DEFAULT_XLOG   = false;    ///< Default flag for enabling x-axis logarithimic scale
    const bool        DEFAULT_YLOG   = false;    ///< Default flag for enabling y-axis logarithimic sclae
    const bool        DEFAULT_LEGEND = false;    ///< Default flag for enabling legend visibility

    // for the legend
    const std::string DEFAULT_LABEL_PREFIX = "data_"; ///< Default legend data name prefix (followed by data_id)
    

    // properties
    std::string name;   ///< File name (no extension)
    std::string title;  ///< Figure title
    std::string xlabel; ///< Figure x-axis label
    std::string ylabel; ///< Figure y-axis label
    bool        grid;   ///< Flag for enabling grid line visibility
    bool        xlog;   ///< Flag for enabling x-axis logarithimic scale
    bool        ylog;   ///< Flag for enabling y-axis logarithimic scale
    bool        legend; ///< Flag for enabling legend visibility

    // data
    std::vector<std::vector<real>> x_data_vec;     ///< Data entries x-component
    std::vector<std::vector<real>> y_data_vec;     ///< Data entries y-component
    std::vector<std::string>       label_name_vec; ///< Data entries label names
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GNU_OUT_H__
