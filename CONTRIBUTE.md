
## Contributing checklist

In terms of syntax, the only rule of thumb is to use descriptive variable names even if they end up being long. Otherwise, any preferred reasonable syntax will be accepted.

However, the we must put an emphasis on code testing:

1. A unit test, integration test, or regression test accompanies the feature. 
This test should automatically fail when the code is erroneously changed.
This mean that we should not `return 0` or copy-paste the tested sections, since changes to the actual code will not affect the outcome of the test.
Tests longer than a few seconds should be tagged as with the suffix MEDIUM, and tests a minute or longer should be tagged with LONG. Long tests are very undesirable and should be avoided when possible.
  * A unit test is often most appropriate, and is aimed at testing a single component of the code. See the test on [Euler's primitive to conservative conversion](https://github.com/dougshidong/PHiLiP/blob/master/tests/unit_tests/euler_unit_test/euler_convert_primitive_conservative.cpp)
  * An integration test runs the entire main program by taking an input file and calling PHiLiP_1/2/3D. It should be derived from the [`TestBase` class](https://github.com/dougshidong/PHiLiP/blob/master/src/testing/tests.h), and have a control file located in the [integration test directory](https://github.com/dougshidong/PHiLiP/tree/master/tests/integration_tests_control_files). Since integrations tests uses multiple components, they usually take longer. Furthermore, the cause of failure is sometimes less obvious. A good suggestion is to use an existing test control file, and only change 1 parameter to help pinpoint issues when it fails.
   * A regression test stores previously computed data to validate future results. Note that this type of test is rarely appropriate since valid changes in the code can fail this type of test. If implemented, a script/code should be made available such that newly computed results can replace the old results. See [file1](https://github.com/dougshidong/PHiLiP/blob/master/tests/unit_tests/regression/jacobian_matrix_regression.cpp) and [file2](https://github.com/dougshidong/PHiLiP/blob/master/tests/unit_tests/regression/matrix_data/copy_matrices.sh)
2. The feature has been documented.
  * Doxygen is currently used to generate documentation. Please visit their [website](http://www.doxygen.nl/manual/docblocks.html) to see how to properly document the code.
  * Function and member variable documentation should be presented in the associated header file. `make doc` should generate a html file in the `/path_to_build/doc/html/index.html` that can be opened used your browser of choice. A non-documented element will generate a warning, which in turn will fail the pull request test.
  * Comments in the .cpp code as appropriate, but prioritize self-documented code by assigning proper variable names.
3. The `master` branch of `https://github.com/dougshidong/PHiLiP` has been merged into your fork and merge conflicts have been resolved.
4. The entire `ctest` suite has been run in `Release` mode and the short/medium length tests have been run in `Debug` mode (using `ctest -E LONG`). Make sure that no tests fails other than the ones listed in the [GitHub issues](https://github.com/dougshidong/PHiLiP/issues?q=is%3Aissue+is%3Aopen+label%3Atestfail) with `testfail` tags.
5. Submit a pull request. Undocumented code will be automatically detected.