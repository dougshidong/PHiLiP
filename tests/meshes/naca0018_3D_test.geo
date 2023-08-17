// airfoil geometry
Include "naca0018.geo";

// geometric parameters
xmax = 150;
ymax = 100;
xV = 51;
xwakeA = 50;
ywakeA = 50;
xwakeB = 25;
ywakeB = 10;
ywakeBL = 5;

radiusA = 30;
radiusB = 0.5;

BL = 0.1;
l_TES = 0.4;

chord = 1;

//--------------------------------------------

/// adjustable parameters

// z axis
spanlength = 0.5; // width of extrusion (5 TES)

n_spanwise_far = 1;
n_spanwise_A = 1;
n_spanwise_B = 1;
n_spanwise_BL = 1;
n_spanwise_TES = 1;

// Mesh ref1

// BL
n_vertical_BL = 2;
r_vertical_BL = 1;

n_inlet_BL = 2;
b_inlet_BL = 1;

n_side_inlet_BL = 2;
r_side_inlet_BL = 1;

n_chordwise_BL = 4;

n_chord_rear_BL = 2;
r_chord_rear_BL = 1;

// TES
n_chordwise_TES = 4;
n_vertical_TES = 2;

// Layer B
n_vertical_B = 2;
r_vertical_B = 0.6;

// Layer A
n_vertical_A = 2;
r_vertical_A = 0.6;

// Farfield
n_vertical_farfield = 1;
r_vertical_farfield = 0.6;


n_inlet = n_inlet_BL + (2 * n_side_inlet_BL) + (2 * n_chordwise_BL) + (2 * n_chord_rear_BL) - 6;

n_near_wake_0 = 8;
r_near_wake_0 = 0.95;
r_TES_wake_0 = 0.7;

n_mid_wake_A = 5;
n_mid_wake_B = 10;

// Wake
n_wake_1 = 2;
r_wake_1 = 0.6;

n_wake_2 = 1;
r_wake_2 = 0.65;

n_wake_3 = 1;
r_wake_3 = 0.8;

n_exit_A = n_vertical_A;
n_exit_far = n_vertical_farfield;


// end parameters


// geometric points
// Farfield
Point(1)={xmax, ymax, 0, pt1};
Point(2)={xV, ymax, 0, pt1};
Point(3)={chord, ymax, 0, pt1};
Point(6)={chord, -ymax, 0, pt1};
Point(7)={xV, -ymax, 0, pt1};
Point(8)={xmax, -ymax, 0, pt1};
// Wake exit points
Point(9)={xmax, -ywakeA, 0, pt1};
Point(10)={xmax, -ywakeBL, 0, pt1};
Point(11)={xmax, ywakeBL, 0, pt1};
Point(12)={xmax, ywakeA, 0, pt1};
// Inner Region A
Point(13)={chord, radiusA, 0, pt1};
Point(16)={chord, -radiusA, 0, pt1};
// Inner Region B
Point(14)={chord, radiusB, 0, pt1};
Point(15)={chord, -radiusB, 0, pt1};
// Boundary Layer Exit
Point(17)={chord, BL, 0, pt1};
Point(18)={chord, -BL, 0, pt1};
// Final wake exit point
Point(21)={xmax, ywakeB, 0, pt1};
Point(22)={xmax, -ywakeB, 0, pt1};
// Added exit points to maintain quads
Point(500) = {xmax + 70, ymax, 0, pt1};
Point(501) = {xmax + 70, ywakeA, 0, pt1};
Point(502) = {xmax + 70, -ywakeA, 0, pt1};
Point(503) = {xmax + 70, -ymax, 0, pt1};
// geometric regions
//+
Line(1003) = {1, 2};
//+
Line(1004) = {2, 3};
//+
Line(1005) = {7, 6};
//+
Line(1006) = {8, 7};
//+
Line(1007) = {503, 9};
//+
Line(1008) = {502, 22};
//+
Line(1009) = {10, 11};
//+
Line(1010) = {501, 21};
//+
Line(1011) = {500, 12};
//+
// A and B Circle
Circle(1022) = {3, 100, 6};
//+
Circle(1023) = {13, 100, 16};
// BL
//+
Point(200)={0.759196,0.045960+BL,0,pt1};
Point(201)={0.516026,0.077852+BL,0,pt1};
Point(202)={0.297608,0.090024+BL,0,pt1};
Point(203)={0.240804,0.088734+BL,0,pt1};
Point(204)={0.188255,0.084968+BL,0,pt1};
Point(205)={0.064341-(0.19*BL),0.059239+(BL),0,pt1};
Point(206)={0.009220-(0.75*BL),0.024586+(0.75*BL),0,pt1};
Point(207)={-BL,0.000000,0,pt1};
Point(208)={0.009220-(0.75*BL),-0.024586-(0.75*BL),0,pt1};
Point(209)={0.064341-(0.19*BL),-0.059239-(BL),0,pt1};
Point(210)={0.188255,-0.084968-BL,0,pt1};
Point(211)={0.240804,-0.088734-BL,0,pt1};
Point(212)={0.297608,-0.090024-BL,0,pt1};
Point(213)={0.516026,-0.077852-BL,0,pt1};
Point(214)={0.759196,-0.045960-BL,0,pt1};
Line(1026)={17,200};
Spline(1027)={200 : 214};
Line(1028)={214, 18};
//+
// Layer B
//+
Point(300)={0.759196,0.045960+radiusB,0,pt1};
Point(301)={0.516026,0.077852+radiusB,0,pt1};
Point(302)={0.297608,0.090024+radiusB,0,pt1};
Point(303)={0.064341-(0.275*radiusB),0.059239+(radiusB),0,pt1};
Point(304)={0.009220-(0.75*radiusB),0.024586+(0.75*radiusB),0,pt1};
Point(305)={-radiusB,0.000000,0,pt1};
Point(306)={0.009220-(0.75*radiusB),-0.024586-(0.75*radiusB),0,pt1};
Point(307)={0.064341-(0.275*radiusB),-0.059239-(radiusB),0,pt1};
Point(308)={0.297608,-0.090024-radiusB,0,pt1};
Point(309)={0.516026,-0.077852-radiusB,0,pt1};
Point(310)={0.759196,-0.045960-radiusB,0,pt1};
Line(1029)={14,300};
Spline(1030)={300 : 310};
Line(1039)={310, 15};
//+
// Vertical Lines
Line(1031) = {3, 13};
//+
Line(1032) = {13, 14};
//+
Line(1033) = {14, 17};
//+
Line(1034) = {17, 101};
//+
Line(1035) = {195, 18};
//+
Line(1036) = {15, 18};
//+
Line(1037) = {16, 15};
//+
Line(1038) = {6, 16};
//+
// Seperating Points
//+
// Layer A
Point(400) = {75, 40, 0, pt1};
//+
Point(401) = {17, 32, 0, pt1};
//+
Point(402) = {75, -40, 0, pt1};
//+
Point(403) = {17, -32, 0, pt1};
//+
// Layer B
Point(404) = {58, 4, 0, pt1};
//+
Point(405) = {27.75, 2.4, 0, pt1};
//+
Point(406) = {1.65, 0.55, 0, pt1};
//+
Point(407) = {58, -4, 0, pt1};
//+
Point(408) = {27.75, -2.4, 0, pt1};
//
Point(409) = {1.65, -0.55, 0, pt1};
//+
// Layer BL
Point(410) = {54, 2, 0, pt1};
//+
Point(411) = {26.2, 1, 0, pt1};
//+
Point(412) = {1.45, 0.115, 0, pt1};
//+
Point(413) = {54, -2, 0, pt1};
//+
Point(414) = {26.2, -1, 0, pt1};
//+
Point(415) = {1.45, -0.115, 0, pt1};
// TES region
Point(416) = {1.11, 0.00189, 0, pt1};
//+
Point(417) = {1.11, -0.00189, 0, pt1};
// TE transition region
Point(418) = {1.2, BL+0.01, 0, pt1};
//+
Point(419) = {1.2, -BL-0.01, 0, pt1};
//
// Lines Seperating Domain
//
// Farfield
//+
Line(1040) = {1, 400};
//+
Line(1041) = {2, 401};
//+
Line(1042) = {7, 403};
//+
Line(1043) = {8, 402};
// Layer A
//+
Line(1044) = {12, 404};
//+
Line(1045) = {400, 405};
//+
Line(1046) = {401, 406};
//+
Line(1047) = {403, 409};
//+
Line(1048) = {402, 408};
//+
Line(1049) = {9, 407};
// Horizontal Lines A
//+
Line(1050) = {12, 400};
//+
Line(1051) = {400, 401};
//+
Line(1052) = {401, 13};
//+
Line(1053) = {403, 16};
//+
Line(1054) = {402, 403};
//+
Line(1055) = {9, 402};
// Horizontal Lines B
//+
Line(1056) = {21, 404};
//+
Line(1057) = {404, 405};
//+
Line(1058) = {405, 406};
//+
Line(1059) = {406, 14};
//+
Line(1060) = {409, 15};
//+
Line(1061) = {408, 409};
//+
Line(1062) = {407, 408};
//+
Line(1063) = {22, 407};
// V Layer B
//+
Line(1064) = {404, 410};
//+
Line(1065) = {405, 411};
//+
Line(1066) = {406, 412};
//+
Line(1067) = {409, 415};
//+
Line(1068) = {408, 414};
//+
Line(1069) = {407, 413};
//Horizontal Layer BL
//+
Line(1070) = {11, 410};
//+
Line(1071) = {410, 411};
//+
Line(1072) = {411, 412};
//+
Line(1073) = {412, 418};
//+
Line(1074) = {18, 419};
//+
Line(1075) = {414, 415};
//+
Line(1076) = {413, 414};
//+
Line(1077) = {10, 413};
// V Layer BL
//+
Line(1084) = {418, 416};
//+
Line(1085) = {417, 419};
//+
// TES region
Line(1086) = {416, 101};
//+
Line(1087) = {195, 417};
//+
Line(1088) = {417, 416};
//+
Line(1112) = {418, 17};
//+
Line(1113) = {415, 419};
//+
// Leading Edge
//+
Line(1089) = {304, 206};
//+
Line(1090) = {206, 145};
//+
Line(1091) = {306, 208};
//+
Line(1092) = {208, 151};
//+
Line(1093) = {303, 205};
//+
Line(1094) = {205, 140};
//+
Line(1095) = {307, 209};
//+
Line(1096) = {209, 156};
//+
Split Curve {1001} Point {140, 145, 151, 156};
//+
Split Curve {1027} Point {205, 206, 208, 209};
//+
Split Curve {1030} Point {303, 304, 306, 307};
// Vertical Wake Lines
//+
Line(1129) = {410, 413};
//+
Line(1130) = {411, 414};
//+
Line(1131) = {412, 415};
//+
Line(1132) = {21, 11};
//+
Line(1133) = {22, 10};
//+
Line(1134) = {418, 419};

// Exit Lines
Line(1135) = {500, 1};
//+
Line(1136) = {501, 12};
//+
Line(1137) = {502, 9};
//+
Line(1138) = {503, 8};
//
// Transfinite Curves
//
// Leading Edge
Transfinite Curve {1116, 1121, 1126} = n_inlet_BL Using Bump b_inlet_BL;
//+
Transfinite Curve {1120, 1115, 1125} = n_side_inlet_BL Using Progression 1 / r_side_inlet_BL;
Transfinite Curve {1117, 1122, 1127} = n_side_inlet_BL Using Progression r_side_inlet_BL;
// Chord
Transfinite Curve {1114, 1119, 1118, 1123, 1124, 1128} = n_chordwise_BL Using Progression 1;
//+
Transfinite Curve {-1026, 1028, -1029, 1039} = n_chord_rear_BL Using Progression r_chord_rear_BL;
//+
Transfinite Curve {1114, 1118} = n_chordwise_BL + n_chord_rear_BL - 1 Using Progression 1;
//+
// Wake
//+
Transfinite Curve {1073, 1113} = n_near_wake_0 + 1- n_chordwise_TES Using Progression r_TES_wake_0;
//+
Transfinite Curve {1112, 1074} = n_chordwise_TES Using Progression 1;
//+
Transfinite Curve {1084, 1034, 1094, 1090, 1092, 1096, 1035, 1085} = n_vertical_BL Using Progression r_vertical_BL;
//+
Transfinite Curve {1075, 1072, 1058, 1051, 1061, 1054, 1003, 1006} = n_wake_1 Using Progression r_wake_1;
//+
Transfinite Curve {1070, 1077, 1056, 1063, 1136, 1137} = n_wake_3 Using Progression r_wake_3;
//+
Transfinite Curve {1022, 1023} = n_inlet Using Progression 1;
//+
Transfinite Curve {1044, 1045, 1046, 1032, 1037, 1047, 1048, 1049} = n_vertical_A Using Progression r_vertical_A;
//+
Transfinite Curve {1010, 1008} = n_exit_A Using Progression r_vertical_A;
//+
Transfinite Curve {1011, 1040, 1041, 1031, 1038, 1042, 1043, 1007} = n_vertical_farfield Using Progression r_vertical_farfield;
//+
Transfinite Curve {1011, 1007} = n_exit_far Using Progression r_vertical_farfield;
// TES
Transfinite Curve {1086, 1087} = n_chordwise_TES Using Progression 1;
//+
Transfinite Curve {1088} = 2 * n_vertical_TES - 1 Using Progression 1;
//+
Transfinite Curve {999, 1002} = n_vertical_TES Using Progression 1;
//+
Transfinite Curve {1059, 1060, 1052, 1053, 1004, 1005} = n_near_wake_0 Using Progression r_near_wake_0;
//+
Transfinite Curve {1071, 1076, 1050, 1057, 1062, 1055, 1135, 1138} = n_wake_2 Using Progression r_wake_2;
//+
Transfinite Curve {1066, 1033, 1093, 1089, 1091, 1095, 1036, 1067, 1065, 1068, 1064, 1069, 1132, 1133} = n_vertical_B Using Progression r_vertical_B;
//+
Transfinite Curve {1130, 1131, 1134, 1129, 1009} = 2 * n_vertical_TES - 1 Using Progression 1;
//
// Planes
//
Curve Loop(1) = {1086, -999, -1002, 1087, 1088};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1086, -1034, -1112, 1084};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {1114, -1094, -1119, -1026, 1034};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {1115, -1090, -1120, 1094};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {1116, -1092, -1121, 1090};
//+
Plane Surface(5) = {5};
//+
Curve Loop(42) = {1134, -1085, 1088, -1084};
//+
Plane Surface(42) = {42};
//+
Curve Loop(6) = {1117, -1096, -1122, 1092};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {1118, 1035, -1028, -1123, 1096};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {1087, 1085, -1074, -1035};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {1073, 1134, -1113, -1131};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {1072, 1131, -1075, -1130};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {1071, 1130, -1076, -1129};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {1070, 1129, -1077, 1009};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {1070, -1064, -1056, 1132};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {1064, 1071, -1065, -1057};
//+
Plane Surface(14) = {14};
//+
Curve Loop(15) = {1065, 1072, -1066, -1058};
//+
Plane Surface(15) = {15};
//+
Curve Loop(16) = {1066, 1073, 1112, -1033, -1059};
//+
Plane Surface(16) = {16};
//+
Curve Loop(17) = {1033, 1026, 1119, -1093, -1124, -1029};
//+
Plane Surface(17) = {17};
//+
Curve Loop(18) = {1093, 1120, -1089, -1125};
//+
Plane Surface(18) = {18};
//+
Curve Loop(19) = {1089, 1121, -1091, -1126};
//+
Plane Surface(19) = {19};
//+
Curve Loop(20) = {1122, -1095, -1127, 1091};
//+
Plane Surface(20) = {20};
//+
Curve Loop(21) = {1123, 1028, -1036, -1039, -1128, 1095};
//+
Plane Surface(21) = {21};
//+
Curve Loop(22) = {-1113, -1067, 1060, 1036, 1074};
//+
Plane Surface(22) = {22};
//+
Curve Loop(23) = {-1075, -1068, 1061, 1067};
//+
Plane Surface(23) = {23};
//+
Curve Loop(24) = {-1076, -1069, 1062, 1068};
//+
Plane Surface(24) = {24};
//+
Curve Loop(25) = {-1077, -1133, 1063, 1069};
//+
Plane Surface(25) = {25};
//+
Curve Loop(26) = {1136, 1044, -1056, -1010};
//+
Plane Surface(26) = {26};
//+
Curve Loop(27) = {1050, 1045, -1057, -1044};
//+
Plane Surface(27) = {27};
//+
Curve Loop(28) = {1051, 1046, -1058, -1045};
//+
Plane Surface(28) = {28};
//+
Curve Loop(29) = {1052, 1032, -1059, -1046};
//+
Plane Surface(29) = {29};
//+
Curve Loop(30) = {1032, 1029, 1124, 1125, 1126, 1127, 1128, 1039, -1037, -1023};
//+
Plane Surface(30) = {30};
//+
Curve Loop(31) = {-1060, -1047, 1053, 1037};
//+
Plane Surface(31) = {31};
//+
Curve Loop(32) = {-1061, -1048, 1054, 1047};
//+
Plane Surface(32) = {32};
//+
Curve Loop(33) = {-1062, -1049, 1055, 1048};
//+
Plane Surface(33) = {33};
//+
Curve Loop(34) = {-1063, -1008, 1137, 1049};
//+
Plane Surface(34) = {34};
//+
Curve Loop(35) = {1135, 1040, -1050, -1011};
//+
Plane Surface(35) = {35};
//+
Curve Loop(36) = {1003, 1041, -1051, -1040};
//+
Plane Surface(36) = {36};
//+
Curve Loop(37) = {1004, 1031, -1052, -1041};
//+
Plane Surface(37) = {37};
//+
Curve Loop(38) = {1022, 1038, -1023, -1031};
//+
Plane Surface(38) = {38};
//+
Curve Loop(39) = {-1053, -1042, 1005, 1038};
//+
Plane Surface(39) = {39};
//+
Curve Loop(40) = {-1054, -1043, 1006, 1042};
//+
Plane Surface(40) = {40};
//+
Curve Loop(41) = {-1055, -1007, 1138, 1043};
//+
Plane Surface(41) = {41};
//
// Transfinite Surfaces
//
//
// Recombine to quads
//
Transfinite Surface {1} = {416, 101, 195, 417};
//+
Transfinite Surface {2} = {418, 17, 101, 416};
//+
Transfinite Surface {3} = {101, 17, 205, 140};
//+
Transfinite Surface {4} = {205, 206, 145, 140};
//+
Transfinite Surface {5} = {145, 206, 208, 151};
//+
Transfinite Surface {6} = {151, 208, 209, 156};
//+
Transfinite Surface {7} = {156, 209, 18, 195};
//+
Transfinite Surface {8} = {417, 195, 18, 419};
//+
Transfinite Surface {16} = {17, 412, 406, 14};
//+
Transfinite Surface {17} = {14, 303, 205, 17};
//+
Transfinite Surface {18} = {303, 304, 206, 205};
//+
Transfinite Surface {19} = {206, 304, 306, 208};
//+
Transfinite Surface {20} = {209, 208, 306, 307};
//+
Transfinite Surface {21} = {18, 209, 307, 15};
//+
Transfinite Surface {22} = {415, 18, 15, 409};
//+
Transfinite Surface {42} = {418, 416, 417, 419};
//+
Transfinite Surface {9} = {412, 418, 419, 415};
//+
Transfinite Surface {10} = {411, 412, 415, 414};
//+
Transfinite Surface {11} = {410, 411, 414, 413};
//+
Transfinite Surface {12} = {11, 410, 413, 10};
//+
Transfinite Surface {13} = {21, 404, 410, 11};
//+
Transfinite Surface {14} = {404, 405, 411, 410};
//+
Transfinite Surface {15} = {405, 406, 412, 411};
//+
Transfinite Surface {23} = {414, 415, 409, 408};
//+
Transfinite Surface {24} = {413, 414, 408, 407};
//+
Transfinite Surface {25} = {10, 413, 407, 22};
//
// Layer A
//+
Transfinite Surface {26} = {12, 404, 21, 501};
//+
Transfinite Surface {27} = {405, 400, 12, 404};
//+
Transfinite Surface {28} = {400, 401, 406, 405};
//+
Transfinite Surface {29} = {401, 13, 14, 406};
//+
Transfinite Surface {30} = {14, 13, 16, 15};
//+
Transfinite Surface {31} = {409, 15, 16, 403};
//+
Transfinite Surface {32} = {408, 409, 403, 402};
//+
Transfinite Surface {33} = {407, 408, 402, 9};
//+
Transfinite Surface {34} = {9, 407, 22, 502};
//
// Farfield
//+
Transfinite Surface {35} = {1, 400, 12, 500};
//+
Transfinite Surface {36} = {1, 2, 401, 400};
//+
Transfinite Surface {37} = {2, 3, 13, 401};
//+
Transfinite Surface {38} = {13, 3, 6, 16};
//+
Transfinite Surface {39} = {403, 16, 6, 7};
//+
Transfinite Surface {40} = {402, 403, 7, 8};
//+
Transfinite Surface {41} = {8, 402, 9, 503};
//
// Recombine 2D
//
//+
Mesh.RecombineAll = 1;
//+
Mesh.RecombinationAlgorithm = 2;
//
// 3D
//
// TES
//Extrude {0, 0, spanlength} {
//	Surface{1}; 
//	Layers {n_spanwise_TES}; 
// 	Recombine;
//}
// BL
//Extrude {0, 0, spanlength} {
//	Surface{2, 3, 4, 5, 6, 7, 8, 42}; 
//	Layers {n_spanwise_BL}; 
//	Recombine;
//}
// Layer B
//Extrude {0, 0, spanlength} {
//	Surface{16, 17, 18, 19, 20, 21, 22, 9}; 
//	Layers {n_spanwise_B}; 
//	Recombine;
//}
// Layer A
//Extrude {0, 0, spanlength} {
//	Surface{28, 29, 30, 31, 32, 23, 10, 15}; 
//	Layers {n_spanwise_A}; 
//	Recombine;
//}
// Farfield
//Extrude {0, 0, spanlength} {
//	Surface{35, 36, 37, 38, 39, 40, 41, 26, 27, 14, 11, 24, 33, 34, 25, 12, 13}; 
//	Layers {n_spanwise_far}; 
//	Recombine;
//}
Extrude {0, 0, spanlength} {
	Surface{1, 2, 3, 4, 5, 6, 7, 8, 42, 16, 17, 18, 19, 20, 21, 22, 9, 28, 29, 30, 31, 32, 23, 10, 15, 35, 36, 37, 38, 39, 40, 41, 26, 27, 14, 11, 24, 33, 34, 25, 12, 13}; 
	Layers {n_spanwise_TES}; 
	Recombine;
}
//
// Recombine 3D
//
Mesh 3;
//+
RecombineMesh;
//+
Mesh.SubdivisionAlgorithm = 1;
//+
Mesh.SecondOrderLinear = 1;
//+
// Coherence Mesh;
//+
RefineMesh;
//+
//
// Physical Surfaces
//
//+
Physical Volume("MeshInterior", 32) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
//+
Physical Surface("Airfoil", 1001) = {1197, 1290, 1223, 1245, 1267, 1152, 1156};
//+
Physical Surface("Farfield", 1004) = {1772, 1794, 1816, 1868, 1838, 1890, 1912, 1908, 2062, 1926, 1784, 1938, 2136, 2114, 2084, 2066};
//+
Physical Surface("SideWall_z0", 2005) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
//+
Physical Surface("SideWall_z1", 2006) = {1785, 1807, 1829, 1851, 1873, 1895, 1917, 1939, 1961, 1579, 1601, 1653, 1675, 1697, 2049, 2071, 2137, 2115, 2093, 1983, 2005, 2027, 1741, 1763, 1719, 1410, 1432, 1454, 1476, 1508, 1378, 1535, 1557, 1187, 1329, 1214, 1236, 1258, 1280, 1307, 1351, 1165};

//Physical Surface("Farfield", 1004) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 1772, 1794, 1816, 1868, 1838, 1890, 1912, 1908, 2062, 1926, 1784, 1938, 2136, 2114, 2084, 2066, 1785, 1807, 1829, 1851, 1873, 1895, 1917, 1939, 1961, 1579, 1601, 1653, 1675, 1697, 2049, 2071, 2137, 2115, 2093, 1983, 2005, 2027, 1741, 1763, 1719, 1410, 1432, 1454, 1476, 1508, 1378, 1535, 1557, 1187, 1329, 1214, 1236, 1258, 1280, 1307, 1351, 1165};
