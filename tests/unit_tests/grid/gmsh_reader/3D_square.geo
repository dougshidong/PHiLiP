//+
SetFactory("OpenCASCADE");
Sphere(1) = {0, 0, 0, 0.5, -Pi/2, Pi/2, 2*Pi};
////+
//Sphere(2) = {0.1, 0.1, 0, 0.25, -Pi/2, Pi/2, 2*Pi};
////+
//Surface Loop(3) = {1};
////+
//Surface Loop(4) = {2};
////+
//Volume(1) = {3, 4};

// Frontal-Delaunay
Mesh.Algorithm = 6;

Mesh.CharacteristicLengthFactor = 3;
Mesh 3;

//// 2 : simple full quad
//// 3 : blossom full quad
//Mesh.RecombinationAlgorithm = 2; // or 3
//RecombineMesh;

// 1 : all quads
// 2 : all hex
Mesh.SubdivisionAlgorithm = 2;
RefineMesh;

Save "3D_square.msh";
