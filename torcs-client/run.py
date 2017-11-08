#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
from pytocl.driver import Driver
from model import Model
import argparse
import sys
from shutil import copyfile
import time as tm
import traceback

if __name__ == '__main__':
    
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    file_out = open('../debug/out.log', 'w')
    file_err = open('../debug/err.log', 'w')
    sys.stdout = file_out
    sys.stderr = file_err
    
    try:
        #model = Model(I=29, O=4, H=10)
        #model.save_to_file('../rnd.param')
    
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
        # parser.add_argument(
        #     '-n',
        #     '--name',
        #     help='Model name.',
        #     type=str
        # )
        parser.add_argument(
            '-o',
            '--output_file',
            help='File where to print results.',
            type=str
        )
        args, _ = parser.parse_known_args()
    
        print(args.parameters_file)
        print(args.output_file)
        #print(args.name)
        
        if args.parameters_file is not None:
            print('NN driver!!!!!!')
            main(MyDriver(args.parameters_file, out_file=args.output_file))#, name=args.name))
        else:
            main(Driver())

    
    except Exception as exc:
        traceback.print_exc()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        file_out.close()
        file_err.close()

        copyfile('../debug/out.log', '../debug/out_{}.log'.format(tm.time()))
        copyfile('../debug/err.log', '../debug/err_{}.log'.format(tm.time()))
        raise

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    file_out.close()
    file_err.close()
    

    
