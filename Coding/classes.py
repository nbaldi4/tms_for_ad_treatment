import numpy as np
import tvb.simulator.lab as tvb
from tvb.simulator.lab import equations, connectivity, patterns
from tvb.basic.neotraits.api import Attr, NArray, Final, List
from tvb.simulator.simulator import Simulator as Simulator_fromTVB


class Simulator_Modified(Simulator_fromTVB):

        def _loop_update_stimulus(self, step, stimulus):
            """Update stimulus values for current time step."""
            if self.stimulus is not None:
                # TODO stim_step != current step
                stim_step = step - (self.current_step + 1)
                #stimulus[self.model.stvar, :, :] = self.stimulus(stim_step).reshape((1, -1, 1)) #THIS IS THE ORIGINAL LINE FROM TVB AS IN THE FILE simulator.py (the lines below the current are not present in the original version)
                if self.surface==None:
                    stimulus[self.model.stvar, :, :] = self.stimulus(stim_step).reshape((1, -1, 1))
                if self.surface!=None:
                    stimulus[self.model.stvar, :, :] = np.append(self.stimulus(stim_step), np.zeros((int(self.connectivity.cortical.shape[0]-np.count_nonzero(self.connectivity.cortical)),1))).reshape((1, -1, 1))

        def _configure_stimuli(self):
            """ Configure the defined Stimuli for this Simulator """
            if self.stimulus is not None:
                if self.surface:
                    # NOTE the region mapping of the stimuli should also include the subcortical areas
                    #self.stimulus.configure_space(region_mapping=self.surface.region_mapping)
                    pass
                else:
                    self.stimulus.configure_space()

        def run(self, **kwds):
            """Convenience method to call the simulator with **kwds and collect output data."""
            import time # this line has been added respect to TVB original file, this is needed otherwise returns error as "time" is undefined...
            ts, xs = [], []
            for _ in self.monitors:
                ts.append([])
                xs.append([])
            wall_time_start = time.time()
            for data in self(**kwds):
                for tl, xl, t_x in zip(ts, xs, data):
                    if t_x is not None:
                        t, x = t_x
                        tl.append(t)
                        xl.append(x)
                        if np.any(np.isnan(x)): # this line and the next 4 lines have been added respect to TVB original file: they break the two "for" nested loops when the last output of variable "x" contains one or more "nan" values
                            break
                else:
                    continue
                break
            elapsed_wall_time = time.time() - wall_time_start
            #self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
            #              elapsed_wall_time * 1e3 / self.simulation_length)
            self.log.info("")
            for i in range(len(ts)):
                ts[i] = np.array(ts[i])
                xs[i] = np.array(xs[i])
            return list(zip(ts, xs)) 

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