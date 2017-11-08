from pytocl.driver import Driver
from pytocl.car import State, Command
from model import *


class MyDriver(Driver):
    
    def __init__(self, parameters_file=None, name=None, out_file=None):
        super(MyDriver, self).__init__()
        
        if parameters_file is not None:
            print(parameters_file)
            self.model = Model(parameters_file=parameters_file)
        else:
            print('NO WEIGHTS FILE')
            self.model = Model(I=29, O=4, H=20)
        
        self.name = name
        self.out_file = out_file

        self.curr_time = 0.0
        self.time = 0.0
        self.distance = 0.0

        print('Driver initialization completed')

        
        
            
    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        print('Drive')
        input = carstate.to_input_array()
        
        self.distance = carstate.distance_raced
        if self.curr_time > carstate.current_lap_time:
            self.time += carstate.last_lap_time
        
        self.curr_time = carstate.current_lap_time
        
        try:
            output = self.model.step(input)
            
            for i in range(len(output)):
                output[i] = max(0, output[i])
            
            print('Out = ' + str(output))
            
            command = Command()
            
            self.accelerate(output[0], output[1], carstate, command)
            self.steer(output[2], output[3], carstate, command)
            
            # self.steer(carstate, 0.0, command)
            #
            # v_x = min(output[0], 300)
            #
            # self.accelerate(carstate, v_x, command)
        
        except OverflowError as err:
            print('--------- OVERFLOW! ---------')
            self.saveResults()
            raise err
        
            

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
    
    
    def accelerate(self, acceleration, brake, carstate, command):
    
        command.accelerator = acceleration
        
        if acceleration > brake and carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        command.brake = brake

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def steer(self, left, right, carstate, command):
        
        command.steering = left - right
        
        
        
    
    def saveResults(self):
        if self.out_file is not None:
            f = open(self.out_file, 'w')
            #f.write("{}: {}, {}".format(self.name, self.distance, self.curr_time))
            f.write("{}, {}".format(self.distance, self.time + self.curr_time))
            f.close()
    
    def on_shutdown(self):
        """
        Server requested driver shutdown.

        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        
        self.saveResults()
        
        
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None