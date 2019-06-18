#ifndef __ASSERT_COMPARE_ARRAY_H__
#define __ASSERT_COMPARE_ARRAY_H__
#include <assert.h>
template<int nstate>
void assert_compare_array (const std::array<double, nstate> &array1, const std::array<double, nstate> &array2, const double scale2, const double tolerance)
{
    for (int s=0; s<nstate; s++) {
        const double diff = std::abs(array1[s] - scale2*array2[s]);
        std::cout
            << "State " << s+1 << " out of " << nstate
            << std::endl
            << "Array 1 = " << array1[s]
            << std::endl
            << "Array 2 = " << array2[s]
            << std::endl
            << "Difference = " << diff
            << std::endl;
        assert(diff < tolerance);
        if(diff > tolerance) throw "Difference too high";
    }
    std::cout << std::endl
              << std::endl
              << std::endl;
}
#endif
