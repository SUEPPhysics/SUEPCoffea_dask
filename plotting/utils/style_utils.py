import re

default_style = {
    "data": {
        "label": "data",
        "color": "black",
        "fmt": "o",
        "linewidth": 2,
        "linestyle": "",
    },
    "data-VRGJ": {
        "label": "data",
        "color": "black",
        "fmt": "^",
        "linewidth": 2,
        "linestyle": "",
    },
    "MC": {
        "label": "MC",
        "color": "slateblue",
        "fmt": "",
        "linewidth": 2,
        "linestyle": "-",
    },
    "MC-VRGJ": {
        "label": "MC",
        "color": "maroon",
        "fmt": "",
        "linewidth": 2,
        "linestyle": "-",
    },
    "QCD": {  # as used in ggf-offline
        "color": "slateblue",
    },
    "QCD_Pt": {
        "label": "QCD",
        "color": "hotpink",
    },
    "QCD_HT": {
        "label": "QCD",
        "color": "mediumvioletred",
    },
    "VVV": {
        "color": "darkorange",
    },
    "VG": {
        "color": "red",
    },
    "VV": {
        "color": "gold",
    },
    "VH": {
        "color": "royalblue",
    },
    "GJets": {
        "label": r"$\gamma$+jets",
        "color": "maroon",
    },
    "WJetsToLNu": {
        "label": r"W+jets $\rightarrow~\ell+\nu$",
        "color": "deepskyblue",
    },
    "DYJetsToLL": {
        "color": "gray",
    },
    "ttX": {
        "color": "palegreen",
    },
    "tt": {
        "color": "green",
    },
    "ST": {
        "color": "seagreen",
    },
    "TTJets": {
        "color": "midnightblue",
    },
    "VGammaToJJGamma": {
        "color": "sienna",
    },
    "ggf-mS125": {
        "color": "cyan",
    },
    "ggf-mS200": {
        "color": "blue",
    },
    "ggf-mS300": {
        "color": "lightseagreen",
    },
    "ggf-mS400": {
        "color": "green",
    },
    "ggf-mS500": {
        "color": "darkgreen",
    },
    "ggf-mS600": {
        "color": "lawngreen",
    },
    "ggf-mS700": {
        "color": "goldenrod",
    },
    "ggf-mS800": {
        "color": "orange",
    },
    "ggf-mS900": {
        "color": "sienna",
    },
    "ggf-mS1000": {
        "color": "red",
    },
    'SUEP-WH-mS125_T2.0_mPhi2.0_generic': {
        "color": "cyan",
        "linewidth": 3,
        "linestyle": "--",
    },
    'SUEP-WH-mS125_T3.0_mPhi3.0_generic': {
        "color": "lime",
        "linewidth": 3,
        "linestyle": "--",
    },
    'SUEP-WH-mS125_T4.0_mPhi4.0_generic': {
        "color": "dimgray",
        "linewidth": 3,
        "linestyle": "--",
    },
    'SUEP-WH-mS125_T8.0_mPhi8.0_generic': {
        "color": "silver",
        "linewidth": 3,
        "linestyle": "--",
    },
}

mA_map = {
    'leptonic':0.5,
    'hadronic':0.7,
    'generic':1.0
}

def getStyle(sample: str) -> dict:

    if any([sample.endswith(f"_{year}") for year in ["2016apv", "2016", "2017", "2018"]]):
        sample = '_'.join(sample.split("_")[:-1])

    if "GluGluToSUEP"  in sample and "mS" in sample:
        sample = sample[sample.find("mS") + 2 :]
        sample = sample.split("_")[0]
        return default_style["ggf-mS" + sample]

    if sample.startswith("SUEP-WH"):
        pattern = r"SUEP-WH-mS(?P<mS>\d+)_T(?P<TD>\d+\.\d+)_mPhi(?P<mPhi>\d+\.\d+)_(?P<mode>\w+)"
        match = re.search(pattern, sample)
        temp = float(match.group("TD"))
        mPhi = float(match.group("mPhi"))
        mode = match.group("mode")
        mA = mA_map[mode]
        _style = default_style.get(sample, {"linewidth": 3, "linestyle": "--"})
        _style["label"] = f"$T_D$={temp} GeV, $m_{{\phi}}$={mPhi} GeV, $m_{{A'}}$={mA} GeV"
        return _style
    
    if sample in default_style.keys():
        return default_style[sample]

    else:
        return {}


def getColor(sample):
    style = getStyle(sample)
    if style:
        return style.get("color", None)
    else:
        return None


def getStyles(samples):
    styles = []
    for sample in samples:
        styles.append(getStyle(sample))
    return styles


def sf(value, error):
    # Calculate the number of significant figures based on the error
    significant_figures = round(-math.log10(error)) + 1

    # Round the value and error to the determined significant figures
    rounded_value = round(value, significant_figures)
    rounded_error = round(error, significant_figures)

    return rounded_value, rounded_error