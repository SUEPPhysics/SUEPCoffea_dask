def GetPSWeights(self, events):
    if self.scouting == 1:
        if len(events.PSweights[0]) == 10:
            self.out_vars["PSWeight_ISR_up"]   = events.PSWeight[:, 0] * events.PSWeight[:, 6]
            self.out_vars["PSWeight_ISR_down"] = events.PSWeight[:, 0] * events.PSWeight[:, 7]
            self.out_vars["PSWeight_FSR_up"]   = events.PSWeight[:, 0] * events.PSWeight[:, 8]
            self.out_vars["PSWeight_FSR_down"] = events.PSWeight[:, 0] * events.PSWeight[:, 9]
        else:
            self.out_vars["PSWeight"] = events.PSWeight[:, 0] 
    else:
        if len(events.PSWeight[0]) == 4:
            self.out_vars["PSWeight_ISR_up"]   = events.PSWeight[:, 0]
            self.out_vars["PSWeight_ISR_down"] = events.PSWeight[:, 2]
            self.out_vars["PSWeight_FSR_up"]   = events.PSWeight[:, 1]
            self.out_vars["PSWeight_FSR_down"] = events.PSWeight[:, 3]
        else:
            self.out_vars["PSWeight"] = events.PSWeight[:, 0]
    return
