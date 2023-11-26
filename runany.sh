#!/bin/bash

#SBATCH --mail-type=NONE                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=log/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=log/%j.err                  # where to store error messages
#SBATCH --cpus-per-task=4
/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID

# exit on errors
set -o errexit
#START=$1
#END=$2
#shift 2
python $@
echo finished at: `date`
exit 0;
