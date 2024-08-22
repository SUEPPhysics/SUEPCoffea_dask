def GetPrefireWeights(events, syst=""):
    if syst == "up":
        return events.L1PreFiringWeight.Up
    if syst == "down":
        return events.L1PreFiringWeight.Dn
    return events.L1PreFiringWeight.Nom
