N = 1000;
Rx=3;
Ry=6;
theta=5.0*Pi/12.0;
oneD = LinSpace(0.0,1.0,N);

epsilon = 1;
For i In {0: #oneD[]-1}
eta = oneD[i];
x = -(Rx - (Rx-1)*epsilon)*Cos(theta*(2*eta-1));
y = (Ry - (Ry-1)*epsilon)*Sin(theta*(2*eta-1));
Point(1+i) = {x, y, 0, 1};
EndFor
BSpline(1) = {1:N};

epsilon = 0;
For i In {0: #oneD[]-1}
eta = oneD[i];
x = -(Rx - (Rx-1)*epsilon)*Cos(theta*(2*eta-1));
y = (Ry - (Ry-1)*epsilon)*Sin(theta*(2*eta-1));
Point(1+i+N) = {x, y, 0, 1};
EndFor

N1 = N+1;
N2 = 2*N;
BSpline(2) = {N1:N2};

eta=0;
For i In {1: #oneD[]-2}
epsilon = oneD[i];
x = -(Rx - (Rx-1)*epsilon)*Cos(theta*(2*eta-1));
y = (Ry - (Ry-1)*epsilon)*Sin(theta*(2*eta-1));
Point(i+2*N) = {x, y, 0, 1};
EndFor

N3 = 2*N+1;
N4 = 3*N-2;

BSpline(3) = {1,N4:N3,N+1};

eta=1;
For i In {1: #oneD[]-2}
epsilon = oneD[i];
x = -(Rx - (Rx-1)*epsilon)*Cos(theta*(2*eta-1));
y = (Ry - (Ry-1)*epsilon)*Sin(theta*(2*eta-1));
Point(i+3*N-2) = {x, y, 0, 1};
EndFor

N5 = 3*N-1;
N6 = 4*N-4;

BSpline(4) = {N2,N5:N6,N};

//+
Curve Loop(1) = {2, 4, -1, 3};
//+
Plane Surface(1) = {1};
//+
Physical Surface("innervol", 5) = {1};
//+
Physical Curve("supersonic_inflow", 1007) = {2};
//+
Physical Curve("supersonic_outflow", 1008) = {3, 4};
//+
Physical Curve("slipwall", 1001) = {1};

n_radial = 19;
n_orthogonal = 6;
factor = 1;
n_radial = n_radial*factor;
n_orthogonal = n_orthogonal*factor;
//+
Transfinite Curve {2, 1} = n_radial Using Progression 1;
//+
Transfinite Curve {-4, 3} = n_orthogonal Using Progression 1.2;
//+
Transfinite Surface {1};

Mesh.RecombineAll = 1;
//+
Mesh.RecombinationAlgorithm = 2;
