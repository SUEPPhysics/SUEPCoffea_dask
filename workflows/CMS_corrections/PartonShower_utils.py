def GetPSWeights(self, events):

    if len(events.PSWeight[0]) == 4:
        self.out_vars["PSWeight_ISR_up"] = events.PSWeight[:, 0]
        self.out_vars["PSWeight_ISR_down"] = events.PSWeight[:, 2]
        self.out_vars["PSWeight_FSR_up"] = events.PSWeight[:, 1]
        self.out_vars["PSWeight_FSR_down"] = events.PSWeight[:, 3]
    else:
        self.out_vars["PSWeight"] = events.PSWeight[:, 0]
    return
