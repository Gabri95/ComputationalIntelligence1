#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
from pytocl.driver import Driver
from model import Model
import argparse

if __name__ == '__main__':
    
    # model = Model(I=77, O=1, H=30)
    # model.save_to_file('../rnd.param')

    parser = argparse.ArgumentParser(
        description='Client for TORCS racing car simulation with SCRC network'
                    ' server.'
    )
    parser.add_argument(
        '-w',
        '--parameters_file',
        help='Model parameters.',
        type=str
    )
    args, _ = parser.parse_known_args()
    
    
    print(args.parameters_file)
    
    if args.parameters_file is not None:
        main(MyDriver(args.parameters_file))
    else:
        main(Driver())
    
    

    
