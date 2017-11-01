from pytocl.driver import Driver
from pytocl.car import State, Command
from model import Model

class MyDriver(Driver):
    
    
    def __init__(self, weights_file):
        super(MyDriver, self).__init__()
        self.model = Model(weights_file)
    
    def __init__(self):
        super(MyDriver, self).__init__()
        self.model = Model(26, 1, 15)
    
    
    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        
        input = carstate.to_input_array()
        
        output = self.model.step(input)
        
        command = Command()
        self.steer(carstate, 0.0, command)

        v_x = output[0]
        
        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command