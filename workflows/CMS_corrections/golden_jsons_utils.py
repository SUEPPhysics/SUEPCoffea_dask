from coffea import lumi_tools


def applyGoldenJSON(self, events):
    if self.era == 2016:
        LumiJSON = lumi_tools.LumiMask(
            "data/GoldenJSON/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
        )
    elif self.era == 2017:
        LumiJSON = lumi_tools.LumiMask(
            "data/GoldenJSON/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
        )
    elif self.era == 2018:
        LumiJSON = lumi_tools.LumiMask(
            "data/GoldenJSON/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"
        )
    else:
        print("No era is defined. Please specify the year")

    events = events[LumiJSON(events.run, events.luminosityBlock)]

    return events
