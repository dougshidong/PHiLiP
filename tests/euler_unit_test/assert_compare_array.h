#ifndef __ASSERT_COMPARE_ARRAY_H__
#define __ASSERT_COMPARE_ARRAY_H__
#include <assert.h>
template<int nstate>
void assert_compare_array (const std::array<double, nstate> &array1, const std::array<double, nstate> &array2, const double scale2, const double tolerance)
{
    for (int s=0; s<nstate; s++) {
        const double diff = std::abs(array1[s] - scale2*array2[s]);
        double max = std::max(std::abs(array1[s]), std::abs(scale2*array2[s]));
        if(max < tolerance) max = 1.0;
        const double rel_diff = diff/max;
        std::cout
            << "State " << s+1 << " out of " << nstate
            << std::endl
            << "Array 1 = " << array1[s]
            << std::endl
            << "Array 2 = " << array2[s]
            << std::endl
            << "Relative difference = " << rel_diff
            << std::endl;
        if(rel_diff > tolerance) {
            std::cout << "Difference too high. rel_diff=" << rel_diff << " and tolerance=" << tolerance << std::endl;
            std::cout << "Failing test..." << std::endl;
            std::abort();
        }
    }
    std::cout << std::endl
              << std::endl
              << std::endl;
}
#endif
