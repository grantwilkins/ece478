Run on Palmetto Clusters:
qsub -I -l select=1:ncpus=1:ngpus=1:gpu_model=v100:mem=6gb,walltime=5:00:00
module load cuda/11.6.2-gcc/9.5.0
nvcc helloworld.cu -o helloworld
./helloworld

===========================================================

Run on School of Computing Virtual Desktop:
https://virtual.computing.clemson.edu
