Required files "job-p1.sh" "Makefile" "Point.cpp" "problem1.cu" "stopwatch.hpp"
Run command "make"
Run command "dos2unix job-p1.sh job-p1.sh"

Run command "sbatch job-p1.sh"

If there is a problem compiling or running try the following:
Run command "module load cuda/6.0.26"

The output cuda:sequential time should be about 1.23:4.076 ms

op.data contains rough fabric surface infomration
fabric.data contains  fabric yarn infomration


Plotting: op.data
Load gnu plot its onw UI. Open this in current directory. or change current directory. You might have to serch this one.
run command in gnu window "splot "op.dat" u 1:2:3 with lines 1:2:3"

Plotting: fabric.data
Load gnu plot its onw UI. Open this in current directory. or change current directory. You might have to serch this one.
run command in gnu window "splot "fabric.data" u 1:2:3 with lines 1:2:3"
