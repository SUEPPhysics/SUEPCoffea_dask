from coffea import processor


class CheckHLTpaths(processor.ProcessorABC):
    def __init__(self) -> None:
        self._accumulator = processor.value_accumulator(float, 0)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        counts = self.accumulator.identity()
        if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
            counts += 1
        return counts

    def postprocess(self, accumulator):
        return accumulator
