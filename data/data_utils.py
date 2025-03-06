import os
import json

def isSampleSignal(sample: str, year: str, path: str = "../data/") -> bool:
    """
    Check the xsections json database to see if a sample is signal or not.
    """
    xsecs_database = f"{path}/xsections_{year}.json"
    with open(xsecs_database) as file:
        MC_xsecs = json.load(file)
        return bool(MC_xsecs[sample]["signal"])


def getXSection(
    dataset: str, year, path: str = "../data/", failOnKeyError: bool = True
) -> float:
    xsection = 1

    xsec_file = f"{path}/xsections_{year}.json"
    with open(xsec_file) as file:
        MC_xsecs = json.load(file)
        try:
            xsection *= MC_xsecs[dataset]["xsec"]
            xsection *= MC_xsecs[dataset]["kr"]
            xsection *= MC_xsecs[dataset]["br"]
        except KeyError:
            logging.warning(
                f"WARNING: I did not find the xsection for {dataset} in {xsec_file}. Check the dataset name and the relevant yaml file."
            )
            if failOnKeyError:
                raise KeyError(f"Could not find xsection for {dataset} in {xsec_file}")
            else:
                return 1

    return xsection


def getLumi(
    era: str,
    analysis: str,
    path: str = os.path.dirname(__file__),
) -> float:
    """
    Open the lumis.json file, and read from analysis, era.
    """

    with open(f"{path}/lumis.json") as f:
        lumis = json.load(f)
        return lumis[analysis][era]


def apply_normalization(plots: dict, norm: float) -> dict:
    if norm > 0.0:
        for plot in list(plots.keys()):
            plots[plot] = plots[plot] * norm
    else:
        logging.warning("Norm is 0, not applying normalization.")
    return plots