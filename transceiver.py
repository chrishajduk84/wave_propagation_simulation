import cmath
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from stage import Stage1D


class WaveformLambda:
    # Defines a waveform using complex exponential form
    # Store position, amplitude
    def __init__(self, ):
        pass

# @dataclass
# class IQSample:
#     I: float
#     Q: float

class Transceiver:
    def __init__(self, position: Tuple[float, float]):
        self.time_ratio = 1000
        self.time_value = 0
        self.position = position
        self.clk_freq = 1e9


    # def periodic_tx(self, iq_samples: List[IQSample], samples_per_second=20e6) -> WaveformLambda:
    #     real_life_seconds_per_sample = self.time_ratio/samples_per_second   # This should adjust dynamically per simulation speed maybe?TODO
    #     for sample in iq_samples:
    #         output = math.cos(2*math.pi*self.clk_freq*self.time_value) * sample.I + math.sin(2*math.pi*self.clk_freq*self.time_value) * sample.Q
    #         time.sleep(real_life_seconds_per_sample)
    #         self._update_clk(1/samples_per_second)
    #         yield output

    def tx_schedule(self, iq_samples: List[complex], mixer_freq=None, sample_rate=20e6) -> List[Tuple[int, complex]]:
        timed_list = []
        picosecond_counter = 0
        for iq in iq_samples:
            timed_list.append((picosecond_counter, iq))
            if mixer_freq is not None:
                # TODO add a for loop here to add all the carrier frequency samples
                pass
            else:
                picosecond_counter += (1/sample_rate)/1e-12     #Convert samples per second into picoseconds per sample


    def tx(self, iq_sample: complex) -> float:
        return iq_sample
        #return math.cos(2*math.pi*self.clk_freq*self.time_value) * iq_sample.I + math.sin(2*math.pi*self.clk_freq*self.time_value) * iq_sample.Q

    def rx(self):
        pass


if __name__ == "__main__":
    t = Transceiver((0,0))
    g = Stage1D()
    samples = [math.sin(i*2*math.pi/1000) + 1j*0 for i in range(1000)]
    transmitted = []
    indexes = []
    for i in range(len(samples)):
        indexes.append(i)
        amplitude = t.tx_schedule(samples[i])
        transmitted.append(amplitude)

    # Plot the real part of the total electric field
    plt.figure(figsize=(10, 6))
    plt.ion()
    for i in range(10000):
        if i < len(samples):
            g.add_sample(samples[i], position=0)

        stage = g.next_step()

        if i >= 5000:
            print(stage[4990:5011])

        plt.cla()
        # plt.plot(x, np.real(E_total), label="Total Electric Field (E_total)", color='b')
        plt.plot(range(len(stage)), np.real(stage), label="Incident Wave (E_i)", color='g', linestyle='--')
        # plt.plot(indexes, np.real(transmitted2), label="Incident Wave2 (E_i)", color='b', linestyle='-')

        plt.xlabel('Position (m)')
        plt.ylabel('Electric Field (V/m)')
        plt.title('Wave Propagation with Reflection and Transmission')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.pause(0.0001)




