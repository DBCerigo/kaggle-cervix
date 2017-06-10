#PBS -N make_submission
#PBS -d ./
source activate venv
python /home/u3760/cervix/bin/make_submission.py
