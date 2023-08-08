import os
import uproot
import argparse

def check_sample(filename):
    if os.path.isfile(filename):
        try:
            file = uproot.open(filename)
            yvals = file['SUEP_nconst_Cluster70'].to_numpy()[0]
            num_events = len(yvals[yvals > 0])
            if num_events < 100:
                return 'low_stats'
            else:
                return 'ok'
        except:
            return 'corrupted'
    else:
        return 'missing'

def main(era):
    input_file = 'filelist/list_full_signal_offline.txt' 
    if era == '2016apv':
        tag = 'July2023_2016apv'
    elif era == '2016':
        tag = 'July2023_2016'
    elif era == '2017':
        tag = 'July2023_2017'
    elif era == '2018':
        tag = 'March2023'
    else:
        raise ValueError("Era not supported.")
    
    plotsDir = '/data/submit/lavezzo/SUEP/outputs/'

    missing_samples = []
    low_stats_samples = []
    with open(input_file, 'r') as file:
        for line in file:
            sample_name = line.strip()
            file_path = f"{plotsDir}{sample_name}_{tag}.root"
            status = check_sample(file_path)
            
            #print(sample_name, status)

            if status == 'missing' or status == 'corrupted':
                missing_samples.append(sample_name)
            elif status == 'low_stats':
                low_stats_samples.append(sample_name)

    #print()
    #print("Missing samples:")
    for s in missing_samples: print(s)
    #print()
    #print("Low stats samples:")
    for s in low_stats_samples: print(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check sample status based on era.')
    parser.add_argument('--era', choices=['2016apv', '2016', '2017', '2018'], required=True, help="Specify the era.")
    args = parser.parse_args()
    main(args.era)