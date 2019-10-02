namespace dealii{

template <int dim, int spacedim = dim>
class FEFieldManifold : public Manifold<dim, spacedim>
{
public:
    //using MappingFEField = MappingFEField<dim,dim,LinearAlgebra::distributed::Vector<double>, DoFHandler<dim>>

    //const MappingFEField mapping;
    //const Mapping<dim> &mapping;
    const MappingFEField<dim,spacedim,LinearAlgebra::distributed::Vector<double>, DoFHandler<dim>> mapping;
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using myTriangulation = dealii::Triangulation<dim>;
#else
    using myTriangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
    const myTriangulation *tria;

    FEFieldManifold(
        const MappingFEField<dim,dim,LinearAlgebra::distributed::Vector<double>, DoFHandler<dim>> &mapping_input,
        //const Mapping<dim> &mapping_input,
        const myTriangulation *tria_input)
        : mapping(mapping_input)
        , tria(tria_input)
    { };

  
    virtual std::unique_ptr<Manifold<dim, spacedim>>
    clone() const override
    { return std_cxx14::make_unique<FEFieldManifold<dim, spacedim>>(mapping, tria); };
  
    virtual Point<spacedim>
    get_intermediate_point(const Point<spacedim> &p1,
                           const Point<spacedim> &p2,
                           const double           w) const override
    {
        std::cout << "get_intermediate_point p1:" << p1 << " p2: " << p2 << " w: " << w << std::endl;
        //bool found_p1 = false;
        //bool found_p2 = false;
        //std::vector<Triangulation::active_cell_iterator> cells_touching_p1, cells_touching_p2;
        std::vector< dealii::TriaActiveIterator < dealii::CellAccessor<dim,spacedim> > > cells_touching_p1, cells_touching_p2;
        dealii::Tensor<1,dim,double> zero_dist;

        // Check if the vertex if within diameter_factor*diameter to the cell center.
        // If it is, then attempt to transform_real_to_unit
        const double inside_tolerance = 1e-12;
        const double diameter_factor2 = 2.0 * 2.0;
        for (auto cell = tria->begin_active(); cell!=tria->end(); ++cell) {
            if (!cell->is_locally_owned()) continue;

            const double diameter = cell->diameter();
            const double acceptable_distance2 = diameter_factor2 * diameter * diameter;

            // Cell center of straight-sided element
            dealii::Point<spacedim> cell_center = cell->center(false, false);
            for (unsigned int iv=0; iv<dealii::GeometryInfo<dim>::vertices_per_cell; ++iv) {
                if (cell_center.distance_square(p1) < acceptable_distance2) {
                    bool is_inside = false;
                    try {
                        const dealii::Point<dim> ref_vertex = mapping.transform_real_to_unit_cell(cell, p1);
                        is_inside = GeometryInfo<dim>::is_inside_unit_cell(ref_vertex, inside_tolerance);
                    //} catch (const Mapping<dim, spacedim>::ExcTransformationFailed &) {
                    } catch (...) {
                    }
                    if (is_inside) cells_touching_p1.push_back(cell);
                }
                if (cell_center.distance_square(p2) < acceptable_distance2) {
                    bool is_inside = false;
                    try {
                        const dealii::Point<dim> ref_vertex = mapping.transform_real_to_unit_cell(cell, p2);
                        is_inside = GeometryInfo<dim>::is_inside_unit_cell(ref_vertex, inside_tolerance);
                    //} catch (const Mapping<dim, spacedim>::ExcTransformationFailed &) {
                    } catch (...) {
                    }
                    if (is_inside) cells_touching_p2.push_back(cell);
                }
            }
        }
        // Find the intersection of cells touching p1 and p2
        std::vector<dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>>> common_cells(cells_touching_p1.size());
        std::set_intersection(cells_touching_p1.begin(), cells_touching_p1.end(),
                              cells_touching_p2.begin(), cells_touching_p2.end(),
                              common_cells.begin());
        std::cout << std::endl;
        std::cout << "Cells touching p1 & p2 "<< std::endl;
        for (auto cell = common_cells.begin(); cell != common_cells.end(); ++cell) {
            std::cout << (*cell)->user_index() << std::endl;
        }
        // Use first cell in the line to evaluate the mapping
        //const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> physical_cell = *(common_cells.begin());
        //const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> physical_cell = common_cells[0];

        // Bring p1 and p2 back to reference coordinate and get the midpoint
        const dealii::Point<dim> p1_ref = mapping.transform_real_to_unit_cell(common_cells[0], p1);
        const dealii::Point<dim> p2_ref = mapping.transform_real_to_unit_cell(common_cells[0], p2);

        const dealii::Point<dim> intermediate_point_ref = (1.0-w) * p1_ref + w * p2_ref;

        const dealii::Point<dim> intermediate_point = mapping.transform_unit_to_real_cell(common_cells[0], intermediate_point_ref);

        std::cout << "New point located at: ";
        for (int d=0;d<dim;d++) {
            std::cout<<intermediate_point[d] << " ";
        }
        std::cout << std::endl;

        return intermediate_point;
    };
  
    //virtual Point< spacedim > get_intermediate_point (const Point< spacedim > &p1, const Point< spacedim > &p2, const double w) const override
    //{
    //};

    //virtual Point< spacedim > get_new_point (const ArrayView< const Point< spacedim >> &surrounding_points, const ArrayView< const double > &weights) const

    //virtual void get_new_points (const ArrayView< const Point< spacedim >> &surrounding_points, const Table< 2, double > &weights, ArrayView< Point< spacedim >> new_points) const

    //virtual Point< spacedim > project_to_manifold (const ArrayView< const Point< spacedim >> &surrounding_points, const Point< spacedim > &candidate) const

    virtual Point< spacedim > get_new_point_on_line (const typename Triangulation< dim, spacedim >::line_iterator &line) const override {
        std::cout << "get_new_point_on_line" << std::endl;
        const int n_points_to_find = 2;
        std::array <dealii::Point<spacedim>, n_points_to_find> points_to_find;
        for (int ip=0; ip<n_points_to_find; ++ip) {
            points_to_find[ip] = line->vertex(ip);
            std::cout << "Looking for Point " << ip+1 << ": " << points_to_find[ip] << std::endl;
        }

        const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> cell_iterator = find_a_cell_given_vertices<n_points_to_find>(points_to_find);
        std::cout << "Found cell: " << cell_iterator << " with vertices: ";
        for (int ip=0; ip<n_points_to_find; ++ip) {
            std::cout << "Vertex " << ip+1 << ": " << cell_iterator->vertex(ip);
        }
        std::cout << std::endl;

        // Bring the points back to reference coordinate and get the midpoint
        dealii::Point<spacedim> new_reference_point;
        for (int ip=0; ip<n_points_to_find; ++ip) {
            new_reference_point += mapping.transform_real_to_unit_cell(cell_iterator, points_to_find[ip]);
        }
        new_reference_point /= n_points_to_find;

        const dealii::Point<dim> new_physical_point = mapping.transform_unit_to_real_cell(cell_iterator, new_reference_point);

        std::cout << "New point located at: ";
        for (int d=0;d<dim;d++) {
            std::cout<<new_physical_point[d] << " ";
        }
        std::cout << std::endl;

        return new_physical_point;
    };

    virtual Point< spacedim > get_new_point_on_quad (const typename Triangulation< dim, spacedim >::quad_iterator &quad) const override
    {
        std::cout << "get_new_point_on_quad" << std::endl;
        const int n_points_to_find = 4;
        std::array <dealii::Point<spacedim>, n_points_to_find> points_to_find;
        for (int ip=0; ip<n_points_to_find; ++ip) {
            points_to_find[ip] = quad->vertex(ip);
            std::cout << "Looking for Point " << ip+1 << ": " << points_to_find[ip] << std::endl;
        }

        const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> cell_iterator = find_a_cell_given_vertices<n_points_to_find>(points_to_find);
        std::cout << "Found cell: " << cell_iterator << " with vertices: ";
        for (int ip=0; ip<n_points_to_find; ++ip) {
            std::cout << "Vertex " << ip+1 << ": ";
            for (int d=0;d<dim;d++) {
                std::cout<<cell_iterator->vertex(ip) << " ";
            }
        }
        std::cout << std::endl;

        // Bring the points back to reference coordinate and get the midpoint
        dealii::Point<spacedim> new_reference_point;
        for (int ip=0; ip<n_points_to_find; ++ip) {
            new_reference_point += mapping.transform_real_to_unit_cell(cell_iterator, points_to_find[ip]);
        }
        new_reference_point /= n_points_to_find;
        std::cout << "New reference point located at: " << new_reference_point << std::endl;

        const dealii::Point<dim> new_physical_point = mapping.transform_unit_to_real_cell(cell_iterator, new_reference_point);
        std::cout << "New physical point located at: " << new_physical_point << std::endl;

        return new_physical_point;
    };

    //virtual Point< spacedim > get_new_point_on_hex (const typename Triangulation< dim, spacedim >::hex_iterator &hex) const override
    //{
    //};
private:
    /** Given some points, this subroutine will return a cell iterator that has vertices at those locations.
     *  There may be many cells that fit the description if given an edge for example. In that case, the 
     *  first cell found to fit the criteria will be returned.
     */
    template<int n_points_to_find>
    dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> find_a_cell_given_vertices(
            const std::array< dealii::Point<spacedim>, n_points_to_find > &points_to_find) const
    {
        // Each array entry contains the list of cells touching vertex 1,2,..,(n_points_to_find) correspondingly
        std::array < std::vector< dealii::TriaActiveIterator < dealii::CellAccessor<dim,spacedim> > >
                        , n_points_to_find >
                    cells_touching_vertices;

        const dealii::Tensor<1,dim,double> zero_dist;
        for (auto cell = tria->begin_active(); cell!=tria->end(); ++cell) {
            if (!cell->is_locally_owned()) continue;
            for (unsigned int iv=0; iv<dealii::GeometryInfo<dim>::vertices_per_cell; ++iv) {
                const dealii::Point<dim> vertex = cell->vertex(iv);

                for (int ip=0; ip<n_points_to_find; ++ip) {
                    const dealii::Tensor<1,dim,double> distance = vertex - points_to_find[ip];
                    if(distance == zero_dist) {
                        cells_touching_vertices[ip].push_back(cell);
                    }
                }
            }
        }
        // Find the intersection of cells touching all the points.
        std::vector<dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>>> common_cells(cells_touching_vertices[0].size());
        for (int iv=0; iv<n_points_to_find-1; ++iv) {
            std::set_intersection(
                cells_touching_vertices[iv].begin(), cells_touching_vertices[iv].end(),
                cells_touching_vertices[iv+1].begin(), cells_touching_vertices[iv+1].end(),
                common_cells.begin());
        }
        // Use first cell in the line to evaluate the mapping
        const dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> cell = common_cells[0];

        return cell;
    }
    
};

}

