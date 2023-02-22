def GetPSWeights(events, output):
    if len(events.PSWeight[0]) == 4:
        output["vars"]["PSWeight_ISR_up"] = events.PSWeight[:, 0]
        output["vars"]["PSWeight_ISR_down"] = events.PSWeight[:, 2]
        output["vars"]["PSWeight_FSR_up"] = events.PSWeight[:, 1]
        output["vars"]["PSWeight_FSR_down"] = events.PSWeight[:, 3]
    else:
        output["vars"]["PSWeight"] = events.PSWeight[:, 0]
    return
