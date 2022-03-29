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

}
}