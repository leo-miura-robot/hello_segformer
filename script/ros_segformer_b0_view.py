#!/usr/bin/env python


from ros_segformer_b0 import main


if __name__ == '__main__':
    main( 
        config='./local_configs/segformer/B0/segformer.b0.512x1024.city.160k.py',
        checkpoint='./checkpoints/lab_mae_sky_iter_160000.pth',
        # imshow=True
    )
