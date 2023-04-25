#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

namespace PHiLiP {

class Optimizer {

public:
// Member functions

    /// Constructor
    Optimizer(_optimization_param);
    
    /// Destructor
    ~Optimizer();
    
    /// Runs fullspace or reduced space optimizer
    void run_optimizer();

    ParlistType get_parlist();

    void check_derivatives;

// Member variables
    optimization_param;

    dg;

};

} // PHiLiP namespace

#endif
