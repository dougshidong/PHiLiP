
## Serial Debugging

Here is a quickstart guide to debugging. It is highly suggested to use gdb and/or valgrind when the program crashes unexpectedly.
The first step is to compile the program in `DEBUG` mode through `CMAKE_BUILD_TYPE=Debug`.

If ctest fails, using `ctest -V -R failing_test_name` will show the command being run.

For a serial run, you may simply use gdb as intended:
```sh
ROOT$ gdb --args commmand_to_launch_test 
GDB$ run (Executes the program. Can re-launch the program if you forgot to put breakpoints.)
```
For example: 
```
gdb --args /home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/bin/PHiLiP_2D "-i" "/home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/tests/advection_implicit/2d_advection_implicit_strong.prm
```


Additional useful commands are:
```sh
GDB$ break dg.cpp:89 (Add a breakpoint in a filename at a line number. Those breakpoints can be added before launching the program.)
GDB$ continue (Continue the program until the next breakpoint or to the end)
GDB$ step (Execute the next step of instructions. It will go into the functions being called)
GDB$ next (Execute the next line of code in the function. Will NOT go into the functions being called)
GDB$ quit
```

### Memory

Memory leaks can be detected using Valgrind's tool `memcheck`. The application must be compiled in `Debug` mode. For example

```
valgrind --leak-check=full --track-origins=yes /home/ddong/Codes/PHiLiP/build_debug/bin/2D_HighOrder_MappingFEField
```

## Parallel debugging

If the error only occurs when using parallelism, you can use the following example command:
```sh
mpirun -np 2 xterm -hold -e gdb -ex 'break MPI_Abort' -ex run --args /home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/bin/PHiLiP_2D "-i" "/home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/tests/advection_implicit/2d_advection_implicit_strong.prm"
```
This launches 2 xterm processes, each of which will launch gdb processes that will run the code and will have a breakpoint when MPI_Abort is encountered.

### Memory
Since no interaction is needed with Valgrind, we don't need xterm anymore. We can simply use
```sh
mpiexec -np 2 valgrind --leak-check=full  --show-reachable=yes --log-file=logfile.%p.log "/home/ddong/Codes/PHiLiP/build_debug/bin/objective_check"
```
to output to multiple logfiles.