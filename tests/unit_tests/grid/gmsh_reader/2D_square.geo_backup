SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 1, 1, 0};

// Frontal-Delaunay for quads
Characteristic Length {:} = 0.8;
Mesh.Algorithm = 8;
Mesh 2;

// 2 : simple full quad
// 3 : blossom full quad
Mesh.RecombinationAlgorithm = 2; // or 3
RecombineMesh;

// // 1 : all quads
// // 2 : all hex
// Mesh.SubdivisionAlgorithm = 1;
// RefineMesh;

// (0: none, 1: optimization, 2: elastic+optimization, 3: elastic, 4: fast curving)
Mesh.ElementOrder = 4;
SetOrder 4;
Mesh.HighOrderOptimize = 2;
OptimizeMesh "HighOrder";

Save "2D_square.msh";
