import numpy as np
import tvb.simulator.lab as tvb
from tvb.simulator.lab import equations, connectivity, patterns
from tvb.basic.neotraits.api import Attr, NArray, Final, List


class JRPSP(tvb.models.JansenRit):
    state_variable_range = Final(
        default={"y0": np.array([-1.0, 1.0]),
                 "y1": np.array([-500.0, 500.0]),
                 "y2": np.array([-50.0, 50.0]),
                 "y3": np.array([-6.0, 6.0]),
                 "y4": np.array([-20.0, 20.0]),
                 "y5": np.array([-500.0, 500.0]),
                 "y6": np.array([-20.0, 20.0]),
                 "y7": np.array([-500.0, 500.0])})

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7"),
        default=("y0", "y1", "y2", "y3"))

    state_variables = tuple('y0 y1 y2 y3 y4 y5 y6 y7'.split())
    _nvar = 8
    cvar = np.array([6], dtype=np.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        dy = np.zeros((8, state_variables.shape[1], 1))
        # TVB's JR is eq 6 only
        dy[:6] = super().dfun(state_variables[:6], coupling, local_coupling)
        # tack on PSP for efferent following eq 8
        # NB with this, only y12 is coupling var for TVB
        y0, y1, y2, y3, y4, y5, y6, y7 = state_variables
        a_d = self.a / 3.0
        sigm_y1_y2 = 2.0 * self.nu_max / (1.0 + np.exp(self.r * (self.v0 - (y1 - y2))))
        dy[6] = y7
        dy[7] = self.A * a_d * sigm_y1_y2 - 2.0 * a_d * y7 - self.a**2 * y6
        return dy


class MultiStimuliRegion(patterns.StimuliRegion):
    connectivity = Attr(connectivity.Connectivity, required=False)
    temporal = Attr(field_type=equations.TemporalApplicableEquation, required=False)
    weight = NArray(required=False)
    
    def __init__(self, *stimuli):
        super(MultiStimuliRegion, self).__init__()
        self.stimuli = stimuli
    def configure_space(self, *args, **kwds):
        [stim.configure_space(*args, **kwds) for stim in self.stimuli]
    def configure_time(self, *args, **kwds):
        [stim.configure_time(*args, **kwds) for stim in self.stimuli]
    def __call__(self, *args, **kwds):
        return np.array([stim(*args, **kwds) for stim in self.stimuli]).sum(axis=0)
    
    
