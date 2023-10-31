ht = 1.0;
hi = 1.4;
he = 1.4;
m = 0.02;

lt = 0.5;
l_i_e = 1.0;

x_t = lt/2;
y_t = ht/2;

y_i = hi/2;
y_e = he/2;

x_i = (y_t - y_i)/m - x_t;
x_e = (y_e - y_t)/m + x_t;


Point(1) = {-x_t, y_t, 0, 1};
Point(2) = {x_t, y_t, 0, 1};
Point(3) = {-x_t, -y_t, 0, 1};
Point(4) = {x_t, -y_t, 0, 1};

Point(5) = {x_i, y_i, 0, 1};
Point(6) = {x_i, -y_i, 0, 1};

Point(7) = {x_e, y_e, 0, 1};
Point(8) = {x_e, -y_e, 0, 1};

x_i_inlet = x_i - l_i_e;
x_e_exit = x_e + l_i_e;

Point(9) = {x_i_inlet, y_i, 0, 1};
Point(10) = {x_i_inlet, -y_i, 0, 1};

Point(11) = {x_e_exit, y_e, 0, 1};
Point(12) = {x_e_exit, -y_e, 0, 1};


 
//+
Line(1) = {9, 5};
//+
Line(2) = {5, 1};
//+
Line(3) = {1, 2};
//+
Line(4) = {2, 7};
//+
Line(5) = {7, 11};
//+
Line(6) = {11, 12};
//+
Line(7) = {12, 8};
//+
Line(8) = {8, 4};
//+
Line(9) = {4, 3};
//+
Line(10) = {3, 6};
//+
Line(11) = {6, 10};
//+
Line(12) = {10, 9};
//+
Line(13) = {5, 6};
//+
Line(14) = {1, 3};
//+
Line(15) = {2, 4};
//+
Line(16) = {7, 8};
//+
Curve Loop(1) = {1, 13, 11, 12};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, 14, 10, -13};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {3, 15, 9, -14};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {4, 16, 8, -15};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {5, 6, 7, -16};
//+
Plane Surface(5) = {5};
//+
Physical Surface("innervol", 17) = {1, 2, 3, 4, 5};
//+
Physical Curve("slipwall", 1001) = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11};
//+
Physical Curve("farfield", 1004) = {12, 6};
//+
//Physical Curve("outflow", 1005) = {6};

n_horizontal = 4;
n_inlet_front = 2;
n_inlet = 4;
n_throat = 2;
n_exit = 4;
n_exit_back = 2;
//+
Transfinite Curve {12, 13, 14, 15, 16, 6} = n_horizontal Using Progression 1;
//+
Transfinite Curve {1, 11} = n_inlet_front Using Progression 1;
//+
Transfinite Curve {2, -10} = n_inlet Using Progression 1;
//+
Transfinite Curve {3, 9} = n_throat Using Progression 1;
//+
Transfinite Curve {-4, 8} = n_exit Using Progression 1;
//+
Transfinite Curve {5, 7} = n_exit_back Using Progression 1;
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {3};
//+
Transfinite Surface {4};
//+
Transfinite Surface {5};


Mesh.RecombineAll = 1;
//+
Mesh.RecombinationAlgorithm = 2;
