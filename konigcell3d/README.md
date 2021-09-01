# KonigCell3D C Source Code


To integrate the `KonigCell3D` library in your code, simply `#include "konigcell3d.h"` and compile `konigcell3d.c` along with the rest of your program.


## Example

The `example.c` file uses the low-level C functions `kc3d_dynamic`, `kc3d_static` and `kc3d_rasterize` to pixellise four particles on a grid, then print it to the terminal. This output can be redirected to a file, then plotted with `plot_example_output.py`.

First compile `example.c` with your compiler of choice; on Linux you might need to link the `math.h` library manually with `-lm`:

```shell
$> gcc example.c konigcell3d.c -lm -Ofast
```

Then run the pixelliser and redirect its output to a file, say `pixels.csv`:

```shell
$> ./a.out > pixels.csv
```

Finally, plot the pixels with the `plot_example_output.py` Python script; supply the file name as the first command line argument:

```shell
$> python plot_example_output.py pixels.csv
```


### Check SIMD Instructions Used

The `KonigCell3D` code has plenty of opportunities for a smart compiler to insert Single Instruction, Multiple Data calculations. You can check them with the `check_simd.sh` bash script; supply the compiled executable as the first command line argument:

```shell
$> sh check_simd.sh a.out
```

This also generates the file `a.out.asm` containing the Assembly code for our example and library. 
