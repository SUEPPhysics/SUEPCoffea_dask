import numpy as np


def GetPSWeights(self, events):
    if self.scouting == 1:
        if len(events.PSweights[0]) >= 9:
            self.out_vars["PSWeight_ISR_up"] = (
                events.PSweights[:, 0] * events.PSweights[:, 6]
            )
            self.out_vars["PSWeight_ISR_down"] = (
                events.PSweights[:, 0] * events.PSweights[:, 7]
            )
            self.out_vars["PSWeight_FSR_up"] = (
                events.PSweights[:, 0] * events.PSweights[:, 8]
            )
            self.out_vars["PSWeight_FSR_down"] = (
                events.PSweights[:, 0] * events.PSweights[:, 9]
            )
        else:
            self.out_vars["PSWeight"] = np.ones(len(events.PSweights))
    else:
        if len(events.PSWeight[0]) == 4:
            self.out_vars["PSWeight_ISR_up"] = events.PSWeight[:, 0]
            self.out_vars["PSWeight_ISR_down"] = events.PSWeight[:, 2]
            self.out_vars["PSWeight_FSR_up"] = events.PSWeight[:, 1]
            self.out_vars["PSWeight_FSR_down"] = events.PSWeight[:, 3]
        else:
            self.out_vars["PSWeight"] = events.PSWeight[:, 0]
    return
