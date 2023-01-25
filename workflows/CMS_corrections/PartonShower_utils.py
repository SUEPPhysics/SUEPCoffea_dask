def GetPSWeights(self, events):

    if len(events.PSWeight[0]) == 4:
        self._accumulator["vars"]["PSWeight_ISR_up"] = events.PSWeight[:, 0]
        self._accumulator["vars"]["PSWeight_ISR_down"] = events.PSWeight[:, 2]
        self._accumulator["vars"]["PSWeight_FSR_up"] = events.PSWeight[:, 1]
        self._accumulator["vars"]["PSWeight_FSR_down"] = events.PSWeight[:, 3]
    else:
        self._accumulator["vars"]["PSWeight"] = events.PSWeight[:, 0]
    return
