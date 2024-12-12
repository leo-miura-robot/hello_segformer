#!/usr/bin/env python


from ros_segformer_b0 import main


if __name__ == '__main__':
    main( 
        config='./local_configs/segformer/B0/segformer.b0.512x1024.city.160k.py',
        checkpoint='./checkpoints/fch_yugo_4_scapes_b0_iter_28000.pth',
        # imshow=True
    )
