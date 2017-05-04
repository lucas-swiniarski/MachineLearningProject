import subprocess

for learning_rate in [0.01, 0.05,.1,.5,]:
    for max_depth in [2,3,4]:
        for n_estimators in [1000]:
            for reg_lambda in [1,1e2,1e4]:

                slurm_text = """#!/bin/bash
#
#SBATCH --job-name=xgboost_{0}_{1}_{2}_{3}
#SBATCH --time=3:00:00
#SBATCH --mem=12GB
#SBATCH --output=jcv_infer_%A.out
#SBATCH --error=jcv_infer_%A.err
#SBATCH --mail-user=jcv312@nyu.edu

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python3/intel/3.5.3

python3 -m pip install -U scikit-learn --user

cd /scratch/jcv312/ML/
cd xgboost; make -j4
cd python-package; sudo python setup.py install
export PYTHONPATH=/scratch/jcv312/ML/xgboost/python-package

cd /scratch/jcv312/ML/

python3 -u train_xgboost.py -l {0} -m {1} -n {2} -r {3} > xgboost_{0}_{1}_{2}_{3}.log
""".format(learning_rate, max_depth, n_estimators, reg_lambda)

                text_file = open("play.slurm", "wb")
                text_file.write("%s" % slurm_text)
                text_file.close()

                subprocess.call("sbatch ./play.slurm", shell=True)
