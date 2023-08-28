Include "airfoil.geo";

farfield = 3.0;
slope_trailing = 0.515804/0.76136;
slope_leading = 3.31794/(2.35822 + 0.0894631);
x_trailing = farfield/slope_trailing + 1.0;
x_leading = farfield/slope_leading - 0.0894631;

A_left = -0.164;
A_right = -0.107;//-0.054;
B_left = 0.0; 
B_right = 0.2;  //0.5 
C_left = 0.0; 
C_right = 0.2;  //0.5
D_right = 1.14;
E_left = 4.5;
E_right = 4.9;
F_left = 5.06;
F_right = 5.288;

// Parameters for transfinite mesh
n_verticallines = 10;
progression_verticallines = 1.5;
n_wake = 5;
progression_wake = 1.5;
n_smalltrailing = 2;
progression_smalltrailing = 1;
n_smallleading = 2;
progression_smallleading = 1;
n_airfoilmid = 6;
progression_airfoilmid = 1;
n_leading = 4;
progression_leading = 2.0;
n_leading_small_horizontal = 6;
progression_leading_small_horizontal = 1.8;


// Points to define line near the shock 
Point(201) = {B_left, farfield, 0, 1.0};
Point(202) = {C_left, -farfield, 0, 1.0};
Point(203) = {E_left, farfield, 0, 1.0};
Point(204) = {F_left, -farfield, 0, 1.0};
Point(205) = {A_left, 0, 0, 1.0};

// Rest of the points
x_left_farfield = -2.0;
x_right = 6;
Point(206) = {x_left_farfield, farfield, 0, 1.0};
Point(207) = {x_left_farfield, -farfield, 0, 1.0};
Point(208) = {x_right, farfield, 0, 1.0};
Point(209) = {x_right, -farfield, 0, 1.0};
Point(210) = {x_right, 0, 0, 1.0};
Point(211) = {x_left_farfield, 0, 0, 1.0};
x_shift_trailing = 0.15;
Point(214) = {D_right, 0, 0, 1.0};
Point(215) = {E_right, farfield, 0, 1.0};
Point(216) = {F_right, -farfield, 0, 1.0};

h = 0.2;
x_top_left = h*(B_left - A_left)/farfield + A_left;
Point(217) = {x_top_left, h, 0, 1.0};
x_bottom_left = A_left - h*(A_left - C_left)/farfield;
Point(218) = {x_bottom_left, -h, 0, 1.0};
x_leading_shifted = 1.5;
Point(219) = {x_leading_shifted, farfield, 0, 1.0};
Point(220) = {x_leading_shifted, -farfield, 0, 1.0};
Point(221) = {x_left_farfield, h, 0, 1.0};
Point(222) = {x_left_farfield, -h, 0, 1.0};

Point(223) = {A_right, 0 , 0 , 1.0};
Point(224) = {B_right, farfield, 0, 1.0};
Point(225) = {C_right, -farfield, 0, 1.0};
x_top_right = h*(B_right - A_right)/farfield + A_right;
Point(226) = {x_top_right, h, 0, 1.0};
x_bottom_right = A_right - h*(A_right - C_right)/farfield;
Point(227) = {x_bottom_right, -h, 0, 1.0};
//+
Split Curve {1} Point {100, 200};

//+
Line(4) = {207, 202};
//+
Line(6) = {204, 216};
//+
Line(7) = {216, 209};
//+
Line(8) = {209, 210};
//+
Line(9) = {210, 208};
//+
Line(10) = {208, 215};
//+
Line(11) = {215, 203};
//+
Line(13) = {201, 206};
//+
Line(16) = {211, 205};
//+
Line(17) = {214, 210};
//+
Line(19) = {200, 214};
//+
Line(22) = {200, 203};
//+
Line(23) = {200, 204};
//+
Line(24) = {214, 215};
//+
Line(25) = {214, 216};
//+
Line(26) = {206, 221};
//+
Line(27) = {221, 211};
//+
Line(28) = {211, 222};
//+
Line(29) = {222, 207};
//+
Line(30) = {221, 217};
//+
Line(31) = {218, 222};
//+
Line(33) = {220, 204};
//+
Line(34) = {203, 219};
//+
Line(36) = {217, 201};
//+
Line(37) = {218, 202};
//+
Line(38) = {219, 80};
//+
Line(39) = {220, 120};
//+
Line(40) = {205, 217};
//+
Line(41) = {205, 218};
//+
Line(42) = {223, 226};
//+
Line(43) = {226, 224};
//+
Line(44) = {227, 223};
//+
Line(45) = {227, 225};
//+
Line(46) = {224, 201};
//+
Line(47) = {219, 224};
//+
Line(48) = {225, 220};
//+
Line(49) = {202, 225};
//+
Line(50) = {205, 223};
//+
Line(51) = {223, 100};
//+
Line(52) = {217, 226};
//+
Line(53) = {226, 80};
//+
Line(54) = {227, 120};
//+
Line(55) = {218, 227};
//+
Split Curve {3} Point {80};
//+
Split Curve {2} Point {120};
//+
Curve Loop(1) = {4, -37, 31, 29};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {49, -45, -55, 37};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {48, 39, -54, 45};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {33, -23, -59, -39};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {6, -25, -19, 23};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {7, 8, -17, 25};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {10, -24, 17, 9};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {11, -22, 19, 24};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {34, 38, -56, 22};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {47, -43, 53, -38};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {46, -36, 52, 43};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {13, 26, 30, 36};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {27, 16, 40, -30};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {28, -31, -41, -16};
//+
Plane Surface(14) = {14};
//+
Curve Loop(15) = {44, -50, 41, 55};
//+
Plane Surface(15) = {15};
//+
Curve Loop(16) = {42, -52, -40, 50};
//+
Plane Surface(16) = {16};

// Surfaces near leading edge.
//+
Curve Loop(17) = {-42, -53, -57, 51};
//+
Plane Surface(17) = {17};
//+
Curve Loop(18) = {-44, -51, -58, 54};
//+
Plane Surface(18) = {18};


//+
Physical Surface("innervol", 60) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
//+
Physical Curve("farfield", 1004) = {4, 49, 48, 33, 6, 7, 8, 9, 10, 11, 34, 47, 46, 13, 26, 27, 28, 29};
//+
Physical Curve("airfoil", 1001) = {56, 57, 58, 59};

//+
Transfinite Curve {9, 24, 22, -38, 43, 36, -26, 29, 37, 45, -39, 23, 25, -8} = n_verticallines Using Progression progression_verticallines;
//+
Transfinite Curve {-10, 17, 7} = n_wake Using Progression progression_wake;
//+
Transfinite Curve {11, 19, 6} = n_smalltrailing Using Progression progression_smalltrailing;
//+
Transfinite Curve {34, 56, 59, 33} = n_airfoilmid Using Progression progression_airfoilmid;
//+
Transfinite Curve {47, 53, 51, 54, 48} = n_smallleading Using Progression progression_smallleading; // nearer to the leading edge.
//+
Transfinite Curve {49, 55, 50, 52, 46} = n_smallleading Using Progression progression_smallleading;
//+
Transfinite Curve {13, -30, -16, 31, -4} = n_leading Using Progression progression_leading;
//+
Transfinite Curve {-27, 40, 42, -57, 58, -44, 41, 28} = n_leading_small_horizontal Using Progression progression_leading_small_horizontal;
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
//+
Transfinite Surface {6};
//+
Transfinite Surface {7};
//+
Transfinite Surface {8};
//+
Transfinite Surface {9};
//+
Transfinite Surface {10};
//+
Transfinite Surface {11};
//+
Transfinite Surface {12};
//+
Transfinite Surface {13};
//+
Transfinite Surface {14};
//+
Transfinite Surface {15};
//+
Transfinite Surface {16};
//+
Transfinite Surface {17};
//+
Transfinite Surface {18};

Mesh.RecombineAll = 1;
//+
Mesh.RecombinationAlgorithm = 2;

Color Red{Surface{1};}
Color Red{Surface{2};}
Color Red{Surface{3};}
Color Red{Surface{4};}
Color Red{Surface{5};}
Color Red{Surface{6};}
Color Red{Surface{7};}
Color Red{Surface{8};}
Color Red{Surface{9};}
Color Red{Surface{10};}
Color Red{Surface{11};}
Color Red{Surface{12};}
Color Red{Surface{13};}
Color Red{Surface{14};}
Color Red{Surface{15};}
Color Red{Surface{16};}
Color Red{Surface{17};}
Color Red{Surface{18};}
