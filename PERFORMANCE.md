
## Performance Investigation

Problems tend to show up in the 3D version if an algorithm has been implemented inefficiently. It is therefore highly recommended that a 3D test accompanies the implemented features. 

### Computational

Computational bottlenecks can be inspected using Valgrind's tool `callgrind`. It is used as such:

```
valgrind --tool=callgrind /home/ddong/Codes/PHiLiP/build_release/bin/2D_RBF_mesh_movement
```

This will result in a `callgrind.out.#####`. A visualizer such as `kcachegrind` (available through `apt`) can then be used to sort through the results. For example:

```kcachegrind callgrind.out.24250```

### Memory

Apart from memory leaks, it is possible that some required allocations demand too much memory. Valgrind also has a tool for this called `massif`. For example:

```valgrind --tool=massif /home/ddong/Codes/PHiLiP/build_debug/bin/3D_RBF_mesh_movement```

will generate a `massif.out.#####` file that can be visualized using `massif-visualizer` (available through `apt`) as

```massif-visualizer massif.out.18580```
