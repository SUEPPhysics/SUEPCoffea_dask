import numpy as np


def GetPSWeights(self, events):
    if self.scouting == 1:
        if len(events.PSweights[0]) >= 9:
            PSWeight_ISR_up = events.PSweights[:, 0] * events.PSweights[:, 6]
            PSWeight_ISR_down = events.PSweights[:, 0] * events.PSweights[:, 7]
            PSWeight_FSR_up = events.PSweights[:, 0] * events.PSweights[:, 8]
            PSWeight_FSR_down = events.PSweights[:, 0] * events.PSweights[:, 9]
            return (
                PSWeight_ISR_up,
                PSWeight_ISR_down,
                PSWeight_FSR_up,
                PSWeight_FSR_down,
            )
        else:
            PSWeight = np.ones(len(events.PSweights))
            return PSWeight
    else:
        if len(events.PSWeight[0]) == 4:
            PSWeight_ISR_up = events.PSWeight[:, 0]
            PSWeight_ISR_down = events.PSWeight[:, 2]
            PSWeight_FSR_up = events.PSWeight[:, 1]
            PSWeight_FSR_down = events.PSWeight[:, 3]
            return (
                PSWeight_ISR_up,
                PSWeight_ISR_down,
                PSWeight_FSR_up,
                PSWeight_FSR_down,
            )
        else:
            PSWeight = events.PSWeight[:, 0]
            return PSWeight
