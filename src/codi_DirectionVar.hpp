#ifndef __CODI_DIRECTION_VAR__
#define __CODI_DIRECTION_VAR__
/**
 * @brief Global namespace for CoDiPack - Code Differentiation Package
 */
namespace codi {

  /**
   * @brief The vector for direction of the forward mode or the reverse mode.
   *
   * The direction is an array with a static size that can be added and multiplied by
   * a scalar value.
   *
   * @tparam Real  The scalar value type that is used by the array.
   * @tparam  dim  The dimension of the array.
   */
  template<typename Real>
  class DirectionVar {
    private:
      std::vector<Real> vec; /**< The data vector with the given dimension */
      size_t dim; /**< Dimension of the data vector. */

    public:

      /**
       * @brief Creates a zero direction.
       */
      CODI_INLINE DirectionVar()
      : vec(0)
      , dim(0)
      { }

      /**
       * @brief Creates a direction with the same value in every component.
       *
       * @param[in] s  The value that is set to all components.
       * @param[in] n  The size of the direction vector.
       */
      CODI_INLINE DirectionVar(const Real& s, const size_t n)
      // : DirectionVar()
      {
        this->resize(n);
        for(size_t i = 0; i < dim; ++i) {
          vec[i] = s;
        }
      }

      /**
       * @brief Copy constructor.
       *
       * @param[in] d  The other direction.
       */
      CODI_INLINE DirectionVar(const DirectionVar<Real>& d)
      // : DirectionVar()
      {
        this->resize(d.size());
        for(size_t i = 0; i < dim; ++i) {
          vec[i] = d.vec[i];
        }
      }

      /**
       * @brief The direction is initialized with the values from the initializer list.
       *
       * If the list is to small, then only the first m elements are set.
       * If the list is to large, then only the first dim elements are set.
       *
       * @param[in] l  The list with the values for the direction.
       */
      CODI_INLINE DirectionVar(std::initializer_list<Real> l)
      // : DirectionVar()
      {
        size_t size = std::min(dim, l.size());
        const Real* array = l.begin(); // this is possible because the standard requires an array storage
        for(size_t i = 0; i < size; ++i) {
          vec[i] = array[i];
        }
      }

      /**
       * @brief Get number of directions.
       *
       * @return The number of directions.
       */
      CODI_INLINE size_t size() const {
        return dim;
      }
      /**
       * @brief Changes the number of directions.
       *
       * @param[in] new_size  The new number of directions.
       */
      CODI_INLINE void resize(size_t new_size) {
        dim = new_size;
        vec.resize(new_size);
      }

      /**
       * @brief Get the i-th element of the direction.
       *
       * No bounds checks are performed.
       *
       * @param[in] i  The index for the vector.
       *
       * @return The element from the vector.
       */
      CODI_INLINE Real& operator[] (const size_t& i) {
        return vec[i];
      }

      /**
       * @brief Get the i-th element of the direction.
       *
       * No bounds checks are performed.
       *
       * @param[in] i  The index for the vector.
       *
       * @return The element from the vector.
       */
      CODI_INLINE const Real& operator[] (const size_t& i) const {
        return vec[i];
      }

      /**
       * @brief Assign operator for the direction.
       *
       * @param[in] v  The values from the direction are set to the values of this direction object.
       *
       * @return Reference to this object.
       */
      CODI_INLINE DirectionVar<Real>& operator = (const DirectionVar<Real>& v) {
        this->resize(v.size());
        for(size_t i = 0; i < dim; ++i) {
          this->vec[i] = v.vec[i];
        }

        return *this;
      }

      /**
       * @brief Update operator for the direction.
       *
       * @param[in] v  The values from the direction are added to the values of this direction object.
       *
       * @return Reference to this object.
       */
      CODI_INLINE DirectionVar<Real>& operator += (const DirectionVar<Real>& v) {

        assert(vec.size() == 0 || vec.size() == v.size());
        if (this->size() == 0) this->resize(v.size());
        
        for(size_t i = 0; i < dim; ++i) {
          this->vec[i] += v.vec[i];
        }

        return *this;
      }

      /**
       * @brief Checks if all entries in the direction are also a total zero.
       *
       * @return true if all entries are a total zero.
       */
      CODI_INLINE bool isTotalZero() const {
        for(size_t i = 0; i < dim; ++i) {
          if( !codi::isTotalZero(vec[i])) {
            return false;
          }
        }

        return true;
      }
  };

  /**
   * @brief Tests if all elements of the given direction are finite.
   *
   * Calls on all elements codi::isfinite.
   *
   * @param[in] d  The direction vector that is tested.
   * @return true if all elements are finite.
   * @tparam Real  The computation type of the direction vector.
   * @tparam  dim  The dimension of the direction vector.
   */
  template<typename Real>
  bool isfinite(const DirectionVar<Real>& d) {
    bool finite = true;

    for(size_t i = 0; i < d.size(); ++i) {
      finite &= codi::isfinite(d[i]);
    }

    return finite;
  }

  /**
   * @brief Scalar multiplication of a direction.
   *
   * Performs the operation w = s * v
   *
   * @param[in] s  The scalar value for the multiplication.
   * @param[in] v  The direction that is multiplied.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator * (const Real& s, const DirectionVar<Real>& v) {
    DirectionVar<Real> r;
    r.resize(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
      r[i] = s * v[i];
    }

    return r;
  }

  /**
   * \copydoc operator*(const Real& s, const DirectionVar<Real>& v)
   */
  template<typename Real, typename = typename std::enable_if<!std::is_same<Real, typename TypeTraits<Real>::PassiveReal>::value>::type>
  CODI_INLINE DirectionVar<Real> operator * (const typename TypeTraits<Real>::PassiveReal& s, const DirectionVar<Real>& v) {
    DirectionVar<Real> r;
    r.resize(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
      r[i] = s * v[i];
    }

    return r;
  }

  /**
   * @brief Scalar multiplication of a direction.
   *
   * Performs the operation w = v * s
   *
   * @param[in] v  The direction that is multiplied.
   * @param[in] s  The scalar value for the multiplication.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator * (const DirectionVar<Real>& v, const Real& s) {
    return s * v;
  }

  /**
   * \copydoc operator*(const DirectionVar<Real>& v, const Real& s)
   */
  template<typename Real, typename , typename = typename std::enable_if<!std::is_same<Real, typename TypeTraits<Real>::PassiveReal>::value>::type>
  CODI_INLINE DirectionVar<Real> operator * (const DirectionVar<Real>& v, const typename TypeTraits<Real>::PassiveReal& s) {
    return s * v;
  }

  /**
   * @brief Scalar division of a direction.
   *
   * Performs the operation w = v / s
   *
   * @param[in] v  The direction that is divided.
   * @param[in] s  The scalar value for the division.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator / (const DirectionVar<Real>& v, const Real& s) {
    DirectionVar<Real> r;
    r.resize(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
      r[i] = v[i] / s;
    }

    return r;
  }

  /**
   * @brief Addition of two directions.
   *
   * Performs the operation w = v1 + v2
   *
   * @param[in] v1  The first direction that is added.
   * @param[in] v2  The second direction that is added.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator + (const DirectionVar<Real>& v1, const DirectionVar<Real>& v2) {
    DirectionVar<Real> r;
    r.resize(v1.size());
    for(size_t i = 0; i < v1.size(); ++i) {
      r[i] = v1[i] + v2[i];
    }

    return r;
  }

  /**
   * @brief Subtraction of two directions.
   *
   * Performs the operation w = v1 - v2
   *
   * @param[in] v1  The first direction that is added.
   * @param[in] v2  The second direction that is subtracted.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator - (const DirectionVar<Real>& v1, const DirectionVar<Real>& v2) {
    DirectionVar<Real> r;
    r.resize(v1.size());
    for(size_t i = 0; i < v1.size(); ++i) {
      r[i] = v1[i] - v2[i];
    }

    return r;
  }

  /**
   * @brief Negation of a direction.
   *
   * Performs the negation on all elements.
   *
   * @param[in] v  The first direction that is added.
   *
   * @return The direction with the result.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  CODI_INLINE DirectionVar<Real> operator - (const DirectionVar<Real>& v) {
    DirectionVar<Real> r;
    r.resize(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
      r[i] = -v[i];
    }

    return r;
  }

  /**
   * @brief Check if at least one component of the direction is not equal to s.
   *
   * The operator returns false if all the components of the direction v are equal to s,
   * true otherwise.
   *
   * @param[in] s  The scalar value that is checked against the components of the direction
   * @param[in] v  The direction that is compared with the scalar value.
   *
   * @return true if at least one component of v is not equal to s, false otherwise.
   *
   * @tparam    A  The type of the scalar value.
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename A, typename Real>
  CODI_INLINE bool operator != (const A& s, const DirectionVar<Real>& v) {
    for(size_t i = 0; i < v.size(); ++i) {
      if( s != v[i] ) {
        return true;
      }
    }

    return false;
  }

  /**
   * @brief Check if at least one component of the direction is not equal to s.
   *
   * The operator returns false if all the components of the direction v are equal to s,
   * true otherwise.
   *
   * @param[in] v  The direction that is compared with the scalar value.
   * @param[in] s  The scalar value that is checked against the components of the direction
   *
   * @return true if at least one component of v is not equal to s, false otherwise.
   *
   * @tparam    A  The type of the scalar value.
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename A, typename Real>
  CODI_INLINE bool operator != (const DirectionVar<Real>& v, const A& s) {
    return s != v;
  }

  /**
   * @brief Output the direction to a stream.
   *
   * The output format is: {v[0], v[1], ..., v[dim - 1]}
   *
   * @param[in,out] os  The output stream that is used for the writing.
   * @param[in]      v  The direction that is written to the stream.
   *
   * @return The output stream os.
   *
   * @tparam Real  The scalar value type that is used by the direction.
   * @tparam  dim  The dimension of the direction.
   */
  template<typename Real>
  std::ostream& operator<<(std::ostream& os, const DirectionVar<Real>& v){
    os << "{";
    for(size_t i = 0; i < v.size(); ++i) {
      if(i != 0) {
        os << ", ";
      }
      os << v[i];
    }
    os << "}";

    return os;
  }
//  /**
//   * @brief Specialization of GradientValueTraits for DirectionVar.
//   */
//  template<typename T>
//  struct GradientValueTraits<DirectionVar<T>> {
//  
//      using Data = T; /**< Entry type of the direction */
//  
//      /**
//       * @brief The array size of the Direction.
//       *
//       * @return n
//       */
//      sCODI_INLINE size_t getVectorSize() {
//        return n;
//      }
//  
//      /**
//       * \copydoc GradientValueTraits::at()
//       */
//      static CODI_INLINE T& at(Direction<T,n>& v, const size_t pos) {
//        return v[pos];
//      }
//  
//      /**
//       * \copydoc GradientValueTraits::at(const T& v, const size_t pos)
//       */
//      static CODI_INLINE const T& at(const Direction<T,n>& v, const size_t pos) {
//        return v[pos];
//      }
//  };

}


