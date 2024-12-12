#!/home/envy/.pyenv/versions/anaconda3-2021.05/envs/segformer_env/bin/python
'''#!/usr/bin/python3.6'''
from ros_segformer_b0 import main


if __name__ == '__main__':
    main( 
        config='./local_configs/segformer/B0/segformer.b0.512x1024.city.160k.py',
        checkpoint='./checkpoints/segformer.b0.512x1024.city.160k.pth',
        imshow=True
    )