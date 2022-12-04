#!/bin/bash
#$ -l h_rt=5:00:00  #time needed
#$ -pe smp 5 #number of cores
#$ -l rmem=10G #number of memery
#$ -o Output/Q2_output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 10g --executor-memory 10g solution.py
