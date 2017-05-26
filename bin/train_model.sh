#PBS -N train_model
#PBS -d ./
source activate test_env
python /home/u3760/cervix/bin/train_model.py
