#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
import argparse




if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser(
    #     description='My AI Client for TORCS racing car simulation with SCRC network'
    #                 ' server.'
    # )
    # parser.add_argument(
    #     '-w',
    #     '--weights',
    #     help='Model parameters.',
    #     type=str
    # )
    # args = parser.parse_args()
    #
    #main(MyDriver(args.w))

    main(MyDriver())
