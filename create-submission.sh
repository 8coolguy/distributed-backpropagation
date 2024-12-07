rm -rf ../project
mkdir ../project
mkdir ../project/serial
mkdir ../project/cuda2
mkdir ../project/openmp
git archive main   | tar -x -C ../project/serial/
git archive cuda2  | tar -x -C ../project/cuda2/
git archive openmp | tar -x -C ../project/openmp/
tar -cvf ../project.tar.gz ../project
