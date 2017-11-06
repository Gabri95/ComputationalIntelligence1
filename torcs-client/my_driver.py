from pytocl.driver import Driver
from pytocl.car import State, Command
from model import Model


class MyDriver(Driver):
    
    def __init__(self, parameters_file=None, **kwargs):
        super(MyDriver, self).__init__()
        
        if parameters_file is not None:
            print(parameters_file)
            self.model = Model(parameters_file=parameters_file)
        else:
            print('NO WEIGHTS FILE')
            self.model = Model(I=77, O=1, H=30)
            
    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        
        input = carstate.to_input_array()
        print('Input len = ' + str(len(input)))
        output = self.model.step(input)
        
        command = Command()
        self.steer(carstate, 0.0, command)

        v_x = min(output[0], 90)
        
        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
