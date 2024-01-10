def GetPrefireWeights(self, events):
    if self.era == "2016" or self.era == "2017" or self.era == "2016apv":
        if self.scouting == 1:
            prefire_nom = events.prefire
            prefire_up = events.prefireup
            prefire_down = events.prefiredown
        else:
            prefire_nom = events.L1PreFiringWeight.Nom
            prefire_up = events.L1PreFiringWeight.Up
            prefire_down = events.L1PreFiringWeight.Dn
    else:
        prefire_nom = 1.0
        prefire_up = 1.0
        prefire_down = 1.0
    return (prefire_nom, prefire_up, prefire_down)
