import awkward as ak
import numpy as np


def track_killing(self, tracks):
    """
    Drop 2.7%, 2.2%, and 2.1% of the tracks randomly at reco-level
    for charged-particles with 1 < pT < 20 GeV in simulation for 2016, 2017, and
    2018, respectively when reclustering the constituents.
     For charged-particles with pT > 20 GeV, 1% of the tracks are dropped randomly
    """

    if self.scouting:
        block1_percent = 0.05
        block2_percent = 0.01
    else:
        year_percent = {"2018": 0.021, "2017": 0.022, "2016": 0.027}
        block1_percent = year_percent[str(self.era)]
        block2_percent = 0.01

    block1_indices = (tracks.pt > 1) & (tracks.pt < 20)
    block2_indices = tracks.pt >= 20

    new_indices = []
    for i in range(len(tracks)):
        event_indices = np.arange(len(tracks[i]))
        event_bool = np.array([True] * len(tracks[i]))

        block1_event_indices = event_indices[block1_indices[i]]
        block1_event_indices_drop = np.random.choice(
            block1_event_indices, int((block1_percent) * len(block1_event_indices))
        )
        event_bool[block1_event_indices_drop] = False

        block2_event_indices = event_indices[block2_indices[i]]
        block2_event_indices_drop = np.random.choice(
            block2_event_indices, int((block2_percent) * len(block2_event_indices))
        )
        event_bool[block2_event_indices_drop] = False

        new_indices.append(list(event_bool))

    new_indices = ak.Array(new_indices)
    tracks = tracks[new_indices]
    return tracks
