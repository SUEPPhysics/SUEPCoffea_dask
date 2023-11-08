import numpy as np


def StorePSWeights(self, events, output):
    if self.scouting == 1:
        if len(events.PSweights[0]) >= 9:
            output["vars"]["PSWeight_ISR_up"] = (
                events.PSweights[:, 0] * events.PSweights[:, 6]
            )
            output["vars"]["PSWeight_ISR_down"] = (
                events.PSweights[:, 0] * events.PSweights[:, 7]
            )
            output["vars"]["PSWeight_FSR_up"] = (
                events.PSweights[:, 0] * events.PSweights[:, 8]
            )
            output["vars"]["PSWeight_FSR_down"] = (
                events.PSweights[:, 0] * events.PSweights[:, 9]
            )
        else:
            output["vars"]["PSWeight"] = np.ones(len(events.PSweights))
    else:
        if len(events.PSWeight[0]) == 4:
            output["vars"]["PSWeight_ISR_up"] = events.PSWeight[:, 0]
            output["vars"]["PSWeight_ISR_down"] = events.PSWeight[:, 2]
            output["vars"]["PSWeight_FSR_up"] = events.PSWeight[:, 1]
            output["vars"]["PSWeight_FSR_down"] = events.PSWeight[:, 3]
        else:
            output["vars"]["PSWeight"] = events.PSWeight[:, 0]
    return
