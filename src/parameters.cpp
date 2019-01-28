#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <list>
#include <iostream>
#include <fstream>
namespace Parameters
{
    using namespace dealii;

    ParameterHandler         prm;
    std::vector<std::string> input_file_names;
    std::string              output_file;
    std::string              output_format;

    struct ManufacturedSolution
    {
        
    }
    void print_usage_message ()
    {
      static const char *message =
          "\n"
          "Converter from deal.II intermediate format to other graphics formats.\n"
          "\n"
          "Usage:\n"
          "    ./step-19 [-p parameter_file] list_of_input_files \n"
          "              [-x output_format] [-o output_file]\n"
          "\n"
          "Parameter sequences in brackets can be omitted if a parameter file is\n"
          "specified on the command line and if it provides values for these\n"
          "missing parameters.\n"
          "\n"
          "The parameter file has the following format and allows the following\n"
          "values (you can cut and paste this and use it for your own parameter\n"
          "file):\n"
          "\n";
      std::cout << message;
      prm.print_parameters (std::cout, ParameterHandler::Text);
    }
}
  // @sect3{Run time parameter handling}

  // Our next job is to define a few classes that will contain run-time
  // parameters (for example solver tolerances, number of iterations,
  // stabilization parameter, and the like). One could do this in the main
  // class, but we separate it from that one to make the program more modular
  // and easier to read: Everything that has to do with run-time parameters
  // will be in the following namespace, whereas the program logic is in the
  // main class.
  //
  // We will split the run-time parameters into a few separate structures,
  // which we will all put into a namespace <code>Parameters</code>. Of these
  // classes, there are a few that group the parameters for individual groups,
  // such as for solvers, mesh refinement, or output. Each of these classes
  // have functions <code>declare_parameters()</code> and
  // <code>parse_parameters()</code> that declare parameter subsections and
  // entries in a ParameterHandler object, and retrieve actual parameter
  // values from such an object, respectively. These classes declare all their
  // parameters in subsections of the ParameterHandler.
  //
  // The final class of the following namespace combines all the previous
  // classes by deriving from them and taking care of a few more entries at
  // the top level of the input file, as well as a few odd other entries in
  // subsections that are too short to warrant a structure by themselves.
  //
  // It is worth pointing out one thing here: None of the classes below have a
  // constructor that would initialize the various member variables. This
  // isn't a problem, however, since we will read all variables declared in
  // these classes from the input file (or indirectly: a ParameterHandler
  // object will read it from there, and we will get the values from this
  // object), and they will be initialized this way. In case a certain
  // variable is not specified at all in the input file, this isn't a problem
  // either: The ParameterHandler class will in this case simply take the
  // default value that was specified when declaring an entry in the
  // <code>declare_parameters()</code> functions of the classes below.
  namespace Parameters
  {

    // @sect4{Parameters::Solver}
    //
    // The first of these classes deals with parameters for the linear inner
    // solver. It offers parameters that indicate which solver to use (GMRES
    // as a solver for general non-symmetric indefinite systems, or a sparse
    // direct solver), the amount of output to be produced, as well as various
    // parameters that tweak the thresholded incomplete LU decomposition
    // (ILUT) that we use as a preconditioner for GMRES.
    //
    // In particular, the ILUT takes the following parameters:
    // - ilut_fill: the number of extra entries to add when forming the ILU
    //   decomposition
    // - ilut_atol, ilut_rtol: When forming the preconditioner, for certain
    //   problems bad conditioning (or just bad luck) can cause the
    //   preconditioner to be very poorly conditioned.  Hence it can help to
    //   add diagonal perturbations to the original matrix and form the
    //   preconditioner for this slightly better matrix.  ATOL is an absolute
    //   perturbation that is added to the diagonal before forming the prec,
    //   and RTOL is a scaling factor $rtol \geq 1$.
    // - ilut_drop: The ILUT will drop any values that have magnitude less
    //   than this value.  This is a way to manage the amount of memory used
    //   by this preconditioner.
    //
    // The meaning of each parameter is also briefly described in the third
    // argument of the ParameterHandler::declare_entry call in
    // <code>declare_parameters()</code>.
    struct Solver
    {
      enum SolverType { gmres, direct };
      SolverType solver;

      enum  OutputType { quiet, verbose };
      OutputType output;

      double linear_residual;
      int max_iterations;

      double ilut_fill;
      double ilut_atol;
      double ilut_rtol;
      double ilut_drop;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };



    void Solver::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("linear solver");
      {
        prm.declare_entry("output", "quiet",
                          Patterns::Selection("quiet|verbose"),
                          "State whether output from solver runs should be printed. "
                          "Choices are <quiet|verbose>.");
        prm.declare_entry("method", "gmres",
                          Patterns::Selection("gmres|direct"),
                          "The kind of solver for the linear system. "
                          "Choices are <gmres|direct>.");
        prm.declare_entry("residual", "1e-10",
                          Patterns::Double(),
                          "Linear solver residual");
        prm.declare_entry("max iters", "300",
                          Patterns::Integer(),
                          "Maximum solver iterations");
        prm.declare_entry("ilut fill", "2",
                          Patterns::Double(),
                          "Ilut preconditioner fill");
        prm.declare_entry("ilut absolute tolerance", "1e-9",
                          Patterns::Double(),
                          "Ilut preconditioner tolerance");
        prm.declare_entry("ilut relative tolerance", "1.1",
                          Patterns::Double(),
                          "Ilut relative tolerance");
        prm.declare_entry("ilut drop tolerance", "1e-10",
                          Patterns::Double(),
                          "Ilut drop tolerance");
      }
      prm.leave_subsection();
    }




    void Solver::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("linear solver");
      {
        const std::string op = prm.get("output");
        if (op == "verbose")
          output = verbose;
        if (op == "quiet")
          output = quiet;

        const std::string sv = prm.get("method");
        if (sv == "direct")
          solver = direct;
        else if (sv == "gmres")
          solver = gmres;

        linear_residual = prm.get_double("residual");
        max_iterations  = prm.get_integer("max iters");
        ilut_fill       = prm.get_double("ilut fill");
        ilut_atol       = prm.get_double("ilut absolute tolerance");
        ilut_rtol       = prm.get_double("ilut relative tolerance");
        ilut_drop       = prm.get_double("ilut drop tolerance");
      }
      prm.leave_subsection();
    }



    // @sect4{Parameters::Refinement}
    //
    // Similarly, here are a few parameters that determine how the mesh is to
    // be refined (and if it is to be refined at all). For what exactly the
    // shock parameters do, see the mesh refinement functions further down.
    struct Refinement
    {
      bool do_refine;
      double shock_val;
      double shock_levels;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };



    void Refinement::declare_parameters (ParameterHandler &prm)
    {

      prm.enter_subsection("refinement");
      {
        prm.declare_entry("refinement", "true",
                          Patterns::Bool(),
                          "Whether to perform mesh refinement or not");
        prm.declare_entry("refinement fraction", "0.1",
                          Patterns::Double(),
                          "Fraction of high refinement");
        prm.declare_entry("unrefinement fraction", "0.1",
                          Patterns::Double(),
                          "Fraction of low unrefinement");
        prm.declare_entry("max elements", "1000000",
                          Patterns::Double(),
                          "maximum number of elements");
        prm.declare_entry("shock value", "4.0",
                          Patterns::Double(),
                          "value for shock indicator");
        prm.declare_entry("shock levels", "3.0",
                          Patterns::Double(),
                          "number of shock refinement levels");
      }
      prm.leave_subsection();
    }


    void Refinement::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("refinement");
      {
        do_refine     = prm.get_bool ("refinement");
        shock_val     = prm.get_double("shock value");
        shock_levels  = prm.get_double("shock levels");
      }
      prm.leave_subsection();
    }



    // @sect4{Parameters::Flux}
    //
    // Next a section on flux modifications to make it more stable. In
    // particular, two options are offered to stabilize the Lax-Friedrichs
    // flux: either choose $\mathbf{H}(\mathbf{a},\mathbf{b},\mathbf{n}) =
    // \frac{1}{2}(\mathbf{F}(\mathbf{a})\cdot \mathbf{n} +
    // \mathbf{F}(\mathbf{b})\cdot \mathbf{n} + \alpha (\mathbf{a} -
    // \mathbf{b}))$ where $\alpha$ is either a fixed number specified in the
    // input file, or where $\alpha$ is a mesh dependent value. In the latter
    // case, it is chosen as $\frac{h}{2\delta T}$ with $h$ the diameter of
    // the face to which the flux is applied, and $\delta T$ the current time
    // step.
    struct Flux
    {
      enum StabilizationKind { constant, mesh_dependent };
      StabilizationKind stabilization_kind;

      double stabilization_value;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };


    void Flux::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("flux");
      {
        prm.declare_entry("stab", "mesh",
                          Patterns::Selection("constant|mesh"),
                          "Whether to use a constant stabilization parameter or "
                          "a mesh-dependent one");
        prm.declare_entry("stab value", "1",
                          Patterns::Double(),
                          "alpha stabilization");
      }
      prm.leave_subsection();
    }


    void Flux::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("flux");
      {
        const std::string stab = prm.get("stab");
        if (stab == "constant")
          stabilization_kind = constant;
        else if (stab == "mesh")
          stabilization_kind = mesh_dependent;
        else
          AssertThrow (false, ExcNotImplemented());

        stabilization_value = prm.get_double("stab value");
      }
      prm.leave_subsection();
    }



    // @sect4{Parameters::Output}
    //
    // Then a section on output parameters. We offer to produce Schlieren
    // plots (the squared gradient of the density, a tool to visualize shock
    // fronts), and a time interval between graphical output in case we don't
    // want an output file every time step.
    struct Output
    {
      bool schlieren_plot;
      double output_step;

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };



    void Output::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("output");
      {
        prm.declare_entry("schlieren plot", "true",
                          Patterns::Bool (),
                          "Whether or not to produce schlieren plots");
        prm.declare_entry("step", "-1",
                          Patterns::Double(),
                          "Output once per this period");
      }
      prm.leave_subsection();
    }



    void Output::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("output");
      {
        schlieren_plot = prm.get_bool("schlieren plot");
        output_step = prm.get_double("step");
      }
      prm.leave_subsection();
    }



    // @sect4{Parameters::AllParameters}
    //
    // Finally the class that brings it all together. It declares a number of
    // parameters itself, mostly ones at the top level of the parameter file
    // as well as several in section too small to warrant their own
    // classes. It also contains everything that is actually space dimension
    // dependent, like initial or boundary conditions.
    //
    // Since this class is derived from all the ones above, the
    // <code>declare_parameters()</code> and <code>parse_parameters()</code>
    // functions call the respective functions of the base classes as well.
    //
    // Note that this class also handles the declaration of initial and
    // boundary conditions specified in the input file. To this end, in both
    // cases, there are entries like "w_0 value" which represent an expression
    // in terms of $x,y,z$ that describe the initial or boundary condition as
    // a formula that will later be parsed by the FunctionParser
    // class. Similar expressions exist for "w_1", "w_2", etc, denoting the
    // <code>dim+2</code> conserved variables of the Euler system. Similarly,
    // we allow up to <code>max_n_boundaries</code> boundary indicators to be
    // used in the input file, and each of these boundary indicators can be
    // associated with an inflow, outflow, or pressure boundary condition,
    // with homogeneous boundary conditions being specified for each
    // component and each boundary indicator separately.
    //
    // The data structure used to store the boundary indicators is a bit
    // complicated. It is an array of <code>max_n_boundaries</code> elements
    // indicating the range of boundary indicators that will be accepted. For
    // each entry in this array, we store a pair of data in the
    // <code>BoundaryCondition</code> structure: first, an array of size
    // <code>n_components</code> that for each component of the solution
    // vector indicates whether it is an inflow, outflow, or other kind of
    // boundary, and second a FunctionParser object that describes all
    // components of the solution vector for this boundary id at once.
    //
    // The <code>BoundaryCondition</code> structure requires a constructor
    // since we need to tell the function parser object at construction time
    // how many vector components it is to describe. This initialization can
    // therefore not wait till we actually set the formulas the FunctionParser
    // object represents later in
    // <code>AllParameters::parse_parameters()</code>
    //
    // For the same reason of having to tell Function objects their vector
    // size at construction time, we have to have a constructor of the
    // <code>AllParameters</code> class that at least initializes the other
    // FunctionParser object, i.e. the one describing initial conditions.
    template <int dim>
    struct AllParameters : public Solver,
      public Refinement,
      public Flux,
      public Output
    {
      static const unsigned int max_n_boundaries = 10;

      struct BoundaryConditions
      {
        typename EulerEquations<dim>::BoundaryKind
        kind[EulerEquations<dim>::n_components];

        FunctionParser<dim> values;

        BoundaryConditions ();
      };


      AllParameters ();

      double diffusion_power;

      double time_step, final_time;
      double theta;
      bool is_stationary;

      std::string mesh_filename;

      FunctionParser<dim> initial_conditions;
      BoundaryConditions  boundary_conditions[max_n_boundaries];

      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
    };



    template <int dim>
    AllParameters<dim>::BoundaryConditions::BoundaryConditions ()
      :
      values (EulerEquations<dim>::n_components)
    {
      for (unsigned int c=0; c<EulerEquations<dim>::n_components; ++c)
        kind[c] = EulerEquations<dim>::no_penetration_boundary;
    }


    template <int dim>
    AllParameters<dim>::AllParameters ()
      :
      diffusion_power(0.),
      time_step(1.),
      final_time(1.),
      theta(.5),
      is_stationary(true),
      initial_conditions (EulerEquations<dim>::n_components)
    {}


    template <int dim>
    void
    AllParameters<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.declare_entry("mesh", "grid.inp",
                        Patterns::Anything(),
                        "intput file name");

      prm.declare_entry("diffusion power", "2.0",
                        Patterns::Double(),
                        "power of mesh size for diffusion");

      prm.enter_subsection("time stepping");
      {
        prm.declare_entry("time step", "0.1",
                          Patterns::Double(0),
                          "simulation time step");
        prm.declare_entry("final time", "10.0",
                          Patterns::Double(0),
                          "simulation end time");
        prm.declare_entry("theta scheme value", "0.5",
                          Patterns::Double(0,1),
                          "value for theta that interpolated between explicit "
                          "Euler (theta=0), Crank-Nicolson (theta=0.5), and "
                          "implicit Euler (theta=1).");
      }
      prm.leave_subsection();


      for (unsigned int b=0; b<max_n_boundaries; ++b)
        {
          prm.enter_subsection("boundary_" +
                               Utilities::int_to_string(b));
          {
            prm.declare_entry("no penetration", "false",
                              Patterns::Bool(),
                              "whether the named boundary allows gas to "
                              "penetrate or is a rigid wall");

            for (unsigned int di=0; di<EulerEquations<dim>::n_components; ++di)
              {
                prm.declare_entry("w_" + Utilities::int_to_string(di),
                                  "outflow",
                                  Patterns::Selection("inflow|outflow|pressure"),
                                  "<inflow|outflow|pressure>");

                prm.declare_entry("w_" + Utilities::int_to_string(di) +
                                  " value", "0.0",
                                  Patterns::Anything(),
                                  "expression in x,y,z");
              }
          }
          prm.leave_subsection();
        }

      prm.enter_subsection("initial condition");
      {
        for (unsigned int di=0; di<EulerEquations<dim>::n_components; ++di)
          prm.declare_entry("w_" + Utilities::int_to_string(di) + " value",
                            "0.0",
                            Patterns::Anything(),
                            "expression in x,y,z");
      }
      prm.leave_subsection();

      Parameters::Solver::declare_parameters (prm);
      Parameters::Refinement::declare_parameters (prm);
      Parameters::Flux::declare_parameters (prm);
      Parameters::Output::declare_parameters (prm);
    }


    template <int dim>
    void
    AllParameters<dim>::parse_parameters (ParameterHandler &prm)
    {
      mesh_filename = prm.get("mesh");
      diffusion_power = prm.get_double("diffusion power");

      prm.enter_subsection("time stepping");
      {
        time_step = prm.get_double("time step");
        if (time_step == 0)
          {
            is_stationary = true;
            time_step = 1.0;
            final_time = 1.0;
          }
        else
          is_stationary = false;

        final_time = prm.get_double("final time");
        theta = prm.get_double("theta scheme value");
      }
      prm.leave_subsection();

      for (unsigned int boundary_id=0; boundary_id<max_n_boundaries;
           ++boundary_id)
        {
          prm.enter_subsection("boundary_" +
                               Utilities::int_to_string(boundary_id));
          {
            std::vector<std::string>
            expressions(EulerEquations<dim>::n_components, "0.0");

            const bool no_penetration = prm.get_bool("no penetration");

            for (unsigned int di=0; di<EulerEquations<dim>::n_components; ++di)
              {
                const std::string boundary_type
                  = prm.get("w_" + Utilities::int_to_string(di));

                if ((di < dim) && (no_penetration == true))
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::no_penetration_boundary;
                else if (boundary_type == "inflow")
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::inflow_boundary;
                else if (boundary_type == "pressure")
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::pressure_boundary;
                else if (boundary_type == "outflow")
                  boundary_conditions[boundary_id].kind[di]
                    = EulerEquations<dim>::outflow_boundary;
                else
                  AssertThrow (false, ExcNotImplemented());

                expressions[di] = prm.get("w_" + Utilities::int_to_string(di) +
                                          " value");
              }

            boundary_conditions[boundary_id].values
            .initialize (FunctionParser<dim>::default_variable_names(),
                         expressions,
                         std::map<std::string, double>());
          }
          prm.leave_subsection();
        }

      prm.enter_subsection("initial condition");
      {
        std::vector<std::string> expressions (EulerEquations<dim>::n_components,
                                              "0.0");
        for (unsigned int di = 0; di < EulerEquations<dim>::n_components; di++)
          expressions[di] = prm.get("w_" + Utilities::int_to_string(di) +
                                    " value");
        initial_conditions.initialize (FunctionParser<dim>::default_variable_names(),
                                       expressions,
                                       std::map<std::string, double>());
      }
      prm.leave_subsection();

      Parameters::Solver::parse_parameters (prm);
      Parameters::Refinement::parse_parameters (prm);
      Parameters::Flux::parse_parameters (prm);
      Parameters::Output::parse_parameters (prm);
    }
  }

