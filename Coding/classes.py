import numpy as np
import tvb.simulator.lab as tvb
from tvb.simulator.lab import equations, connectivity, patterns
from tvb.basic.neotraits.api import Attr, NArray, Final, List


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
        n = state_variables.shape[1]
        dy = np.zeros((8, n, 1))

        y0, y1, y2, y3, y4, y5, y6, y7 = state_variables
        lrc = coupling[0, :]
        short_range_coupling = local_coupling*(y1 - y2)

        # --- CORREZIONE: tutte colonne (n,1) ---
        a_d = np.asarray(self.a).reshape(-1,1) / 3.0
        a_d_2 = np.asarray(self.a).reshape(-1,1)
        b_d = np.asarray(self.b).reshape(-1,1)
        J_d = np.asarray(self.J).reshape(-1,1)
        sigm_y1_y2 = 2 * self.nu_max / (1 + np.exp(self.r * (self.v0 - (y1 - y2))))    
        sigm_y1_y2 = sigm_y1_y2.reshape(-1,1)
        sigm_y0_1  = 2 * self.nu_max / (1 + np.exp(self.r * (self.v0 - (self.a_1 * J_d * y0))))
        sigm_y0_1 = sigm_y0_1.reshape(-1,1)
        sigm_y0_3  = 2 * self.nu_max / (1 + np.exp(self.r * (self.v0 - (self.a_3 * J_d * y0))))
        sigm_y0_3 = sigm_y0_3.reshape(-1,1)

        # ora tutte le variabili hanno shape coerente
        dy[0] = y3
        dy[1] = y4
        dy[2] = y5
        dy[3] = self.A * a_d_2 * sigm_y1_y2 - 2.0 * a_d_2 * y3 - a_d_2 ** 2 * y0
        dy[4] = self.A * a_d_2 * (self.mu + self.a_2 * J_d * sigm_y0_1 + lrc + short_range_coupling) - 2.0 * a_d_2 * y4 - a_d_2 ** 2 * y1
        dy[5] = self.B * b_d * (self.a_4 * J_d * sigm_y0_3) - 2.0 * b_d * y5 - b_d ** 2 * y2
        dy[6] = y7
        dy[7] = self.A * a_d * sigm_y1_y2 - 2*a_d*y7 - (a_d_2**2)*y6

        return dy