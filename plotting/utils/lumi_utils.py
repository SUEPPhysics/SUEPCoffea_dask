import json
import sys
sys.path.append("../")
sys.path.append("../../")
from data.data_utils import getLumi

def lumiLabelWH(year):
    return lumiLabel("WH", year)

def lumiLabel(analysis, year):
    lumi = getLumi(analysis=analysis, era=year)
    if year in ["2017", "2018"]:
        return round(lumi / 1000, 1)
    elif year == "2016":
        apv = year+"apv"
        return round((lumi + getLumi(analysis=analysis, era=apv)) / 1000, 1)
    elif year == "all":
        return round(lumi / 1000, 1)
