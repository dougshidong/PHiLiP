#include "delaunay.h"

namespace PHiLiP {
namespace ProperOrthogonalDecomposition {

Edge::Edge(RowVector2d A, RowVector2d B)
        : node1(A)
        , node2(B)
{}

bool Edge::operator==(const Edge &other) {
    return ((other.node1 == node1 && other.node2 == node2) ||
            (other.node1 == node2 && other.node2 == node1));
}

Triangle::Triangle(RowVector2d A, RowVector2d B, RowVector2d C)
{
    //Formula for coordinates of the circumcenter
    double D = 2*(A(0)*(B(1) - C(1)) + B(0)*(C(1)-A(1)) + C(0)*(A(1)-B(1)));
    double center_x = (1/D)*((std::pow(A(0), 2) + std::pow(A(1), 2))*(B(1) - C(1)) + (std::pow(B(0), 2) + std::pow(B(1), 2))*(C(1) - A(1)) + (std::pow(C(0), 2) + std::pow(C(1), 2))*(A(1) - B(1)));
    double center_y = (1/D)*((std::pow(A(0), 2) + std::pow(A(1), 2))*(C(0) - B(0)) + (std::pow(B(0), 2) + std::pow(B(1), 2))*(A(0) - C(0)) + (std::pow(C(0), 2) + std::pow(C(1), 2))*(B(0) - A(0)));
    circumcenter = {center_x, center_y};
    radius_squared = std::pow(A(0) - center_x, 2) + std::pow(A(1) - center_y, 2);
    edges = {Edge(A,B), Edge(B,C), Edge(A,C)};
    nodes = {A, B, C};
}


Delaunay::Delaunay(Eigen::Matrix<double, Eigen::Dynamic, 2> points)
{
    triangulate(points);
    compute_centroids();
    compute_edge_midpoints();
}

void Delaunay::triangulate(Eigen::Matrix<double, Eigen::Dynamic, 2> points)
{
    RowVector2d min = points.colwise().minCoeff();
    RowVector2d max = points.colwise().maxCoeff();

    double dx = max(0) - min(0);
    double dy = max(0) - min(0);
    const auto dmax = std::max(dx, dy);
    const auto midx = (min(0) + max(0))/2;
    const auto midy = (min(1) + max(1))/2;

    //Make supertriangle ridiculously large to make sure nothing gets cut out
    RowVector2d super_node_1(midx - 1000 * dmax, midy - dmax);
    RowVector2d super_node_2(midx, midy + 1000 * dmax);
    RowVector2d super_node_3(midx + 1000 * dmax, midy - dmax);

    Triangle supertriangle = Triangle(super_node_1, super_node_2, super_node_3);
    triangulation.emplace_back(supertriangle);

    for(auto point : points.rowwise()){
        std::vector<Triangle> triangles;
        std::vector<Edge> bad_edges;

        for(const auto& tri : triangulation){
            //Check if point is inside the circumcircle
            double dist_squared = std::pow(tri.circumcenter(0) - point(0), 2) + std::pow(tri.circumcenter(1) - point(1), 2);
            if((dist_squared - tri.radius_squared) <= 1E-04){
                for(auto& edge : tri.edges){
                    bad_edges.push_back(edge);
                }
            }
            else{
                triangles.push_back(tri);
            }
        }

        // Delete duplicate edges
        std::vector<bool> remove(bad_edges.size(), false);
        for (auto it1 = bad_edges.begin(); it1 != bad_edges.end(); ++it1) {
            for (auto it2 = bad_edges.begin(); it2 != bad_edges.end(); ++it2) {
                if (it1 == it2) {
                    continue;
                }
                if (*it1 == *it2) {
                    remove[std::distance(bad_edges.begin(), it1)] = true;
                    remove[std::distance(bad_edges.begin(), it2)] = true;
                }
            }
        }

        bad_edges.erase(std::remove_if(bad_edges.begin(), bad_edges.end(),[&](auto const& e) { return remove[&e - &bad_edges[0]]; }), bad_edges.end());

        // Update triangulation
        for (auto const& e : bad_edges) {
            triangles.emplace_back(e.node1, e.node2, point);
        }
        triangulation = triangles;
    }

    triangulation.erase(std::remove_if(triangulation.begin(), triangulation.end(),[&](auto const& tri)
    {
        for(auto& node : tri.nodes){
            for(auto& supernode : supertriangle.nodes){
                if(node == supernode){
                    return true;
                }
            }
        }return false;
    }),triangulation.end());
}

void Delaunay::compute_centroids()
{
    centroids.resize(triangulation.size(), 2);
    int row_idx = 0;
    for(auto& triangle : triangulation){
        std::cout << "computing centroid" << std::endl;
        RowVector2d centroid(0,0);
        for(auto& node : triangle.nodes){
            centroid = centroid + node/3;
        }
        centroids.row(row_idx) = centroid;
        row_idx++;
    }
}

void Delaunay::compute_edge_midpoints()
{
    std::vector<RowVector2d> edge_midpoints;

    for(auto& triangle : triangulation){
        for(auto& edge : triangle.edges){
            RowVector2d midpoint = {(edge.node1(0) + edge.node2(0))/2, (edge.node1(1) + edge.node2(1))/2};
            std::cout << midpoint << std::endl;

            if(std::find(edge_midpoints.begin(), edge_midpoints.end(), midpoint) != edge_midpoints.end()) {
                continue;
            } else {
                edge_midpoints.emplace_back(midpoint);
            }
        }
    }
    midpoints.resize(edge_midpoints.size(), 2);
    for(unsigned int i = 0 ; i < edge_midpoints.size() ; i++){
        midpoints.row(i) = edge_midpoints[i];
    }
}

void Delaunay::split_triangle(RowVector2d new_node)
{
    std::vector<Triangle> triangles;

    for(auto& tri : triangulation){

        //Check if new node is inside the triangle
        if(in_triangle(new_node, tri)){
            for(auto& edge : tri.edges){
                if(!on_edge(new_node, edge)){ //make sure not to create a triangle from 3 points on the same line
                    std::cout << "create new triangle" << std::endl;
                    Triangle triangle(edge.node1, edge.node2, new_node);
                    triangles.emplace_back(triangle);
                    for(auto& n : triangle.nodes){
                        std::cout << "Node: " << n(0) << " " << n(1) << std::endl;
                    }
                    //Update midpoints
                    for(auto& e : triangle.edges){
                        std::cout << "new edge" << std::endl;
                        midpoints.conservativeResize(midpoints.rows()+1, 2);
                        RowVector2d midpoint = {(e.node1(0) + e.node2(0))/2, (e.node1(1) + e.node2(1))/2};
                        midpoints.row(midpoints.rows()-1) = midpoint;
                        /*
                        if(edge_has_midpoint(e)){
                            std::cout << "edge already has a point on it" << std::endl;
                        }
                        else{
                            std::cout << "edge has no midpoint on it" << std::endl;
                            midpoints.conservativeResize(midpoints.rows()+1, 2);
                            RowVector2d midpoint = {(e.node1(0) + e.node2(0))/2, (e.node1(1) + e.node2(1))/2};
                            midpoints.row(midpoints.rows()-1) = midpoint;

                        }
                         */
                    }
                    //triangles.emplace_back(edge.node1, edge.node2, new_node);
                }
            }
        }
        else{
            std::cout << "old triangle added" << std::endl;
            triangles.emplace_back(tri);
        }
    }

    triangulation = triangles;
    //compute_centroids();
    //compute_edge_midpoints();
}

bool Delaunay::in_triangle(RowVector2d new_node, Triangle triangle){
    double x1 = triangle.nodes[0](0);
    double x2 = triangle.nodes[1](0);
    double x3 = triangle.nodes[2](0);
    double y1 = triangle.nodes[0](1);
    double y2 = triangle.nodes[1](1);
    double y3 = triangle.nodes[2](1);
    double x = new_node(0);
    double y = new_node(1);

    /* Calculate area of triangle */
    double A = abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2.0);
    /* Calculate area of first subtriangle */
    double A1 = abs((x*(y2-y3) + x2*(y3-y)+ x3*(y-y2))/2.0);
    /* Calculate area of second subtriangle */
    double A2 = abs((x1*(y-y3) + x*(y3-y1)+ x3*(y1-y))/2.0);
    /* Calculate area of third subtriangle */
    double A3 = abs((x1*(y2-y) + x2*(y-y1)+ x*(y1-y2))/2.0);
    /* Check if sum of A1, A2 and A3 is same as A */
    std::cout << "area A: " << A << " area A1: " << A1 << " area A2: " << A2 << " area A3: " << A3 << std::endl;
    return ((A1 + A2 + A3) - A < 1E-08);
}

bool Delaunay::on_edge(RowVector2d new_node, Edge edge){

    double x1 = edge.node1(0);
    double x2 = edge.node2(0);
    double y1 = edge.node1(1);
    double y2 = edge.node2(1);
    double x = new_node(0);
    double y = new_node(1);

    double len = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
    double len1 = std::sqrt(std::pow(x2 - x, 2) + std::pow(y2 - y, 2));
    double len2 = std::sqrt(std::pow(x1 - x, 2) + std::pow(y1 - y, 2));

    if((len1 + len2) - len < 1E-08){
        return true;
    }
    else{
        return false;
    }
}

bool Delaunay::edge_has_midpoint(Edge edge){

    for(auto point : midpoints.rowwise()) {
        double x1 = edge.node1(0);
        double x2 = edge.node2(0);
        double y1 = edge.node1(1);
        double y2 = edge.node2(1);
        double x = point(0);
        double y = point(1);

        double len = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
        double len1 = std::sqrt(std::pow(x2 - x, 2) + std::pow(y2 - y, 2));
        double len2 = std::sqrt(std::pow(x1 - x, 2) + std::pow(y1 - y, 2));

        if ((len1 + len2) - len < 1E-08) {
            return true;
        }
    }
    return false;
}


}
}