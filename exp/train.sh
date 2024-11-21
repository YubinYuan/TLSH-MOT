cd src
python3 train.py --task mot \
                 --exp_id 'bb_6' \
                 --batch_size 32 \
                 --load_model '../models/tlsh_mot.pth'\
                 --data_cfg '../src/lib/cfg/visdrone.json'\
                 --gpus '0,1,2,3'\
                 --lr_step '40'\
                 --lr 9e-5 \
                 --num_epochs 70
cd ..
