import os
import pandas as pd 
import numpy as np
from tqdm import tqdm
from loguru import logger

noise_rate = ['0.10', '0.50', '0.01']
network_version = ['', 'v2']
network_type = ['s', 'm', 'l', 'x']
exp_id = ['1', '2', '3', '4', '5']

if __name__ == "__main__":
    results = None
    results_col = ["acc", "prec", "recall", "f1", "psnr", "cnr", "var"] * 5
    results_row = []
    
    # bandpass streaknet 
    logger.info("Start collecting [bandpass streaknet]...")
    pbar = tqdm(total=len(network_version)*len(network_type)*len(exp_id))
    for n_version in network_version:
        for n_type in network_type:
            for eid in exp_id:
                path = os.path.join("StreakNet_outputs", "streaknet{}_{}_{}".format(n_version, n_type, eid), "bandpass_log.xlsx")
                df = pd.read_excel(path)
                if results is None:
                    results = df.values 
                else:
                    results = np.concatenate([results, df.values], 0)
                results_row.append("bandpass_streaknet{}_{}_{}".format(n_version, n_type, eid))
                pbar.update()
            
            tmp_results = results[-len(exp_id):]
            tmp_results = np.mean(tmp_results, axis=0, keepdims=True)
            results = np.concatenate([results, tmp_results], 0)
            results_row.append("bandpass_streaknet{}_{}_mean".format(n_version, n_type))
            
            tmp_results = np.zeros_like(tmp_results)
            results = np.concatenate([results, tmp_results], 0)
            results_row.append("")
    
    # reconstruction
    logger.info("Start collecting [reconstruction]...")
    pbar = tqdm(total=len(network_version)*len(network_type)*len(exp_id))
    for n_version in network_version:
        for n_type in network_type:
            for eid in exp_id:
                path = os.path.join("StreakNet_outputs", "streaknet{}_{}_{}".format(n_version, n_type, eid), "reconstruction_log.xlsx")
                df = pd.read_excel(path)
                if results is None:
                    results = df.values 
                else:
                    results = np.concatenate([results, df.values], 0)
                results_row.append("streaknet{}_{}_{}".format(n_version, n_type, eid))
                pbar.update()
            
            tmp_results = results[-len(exp_id):]
            tmp_results = np.mean(tmp_results, axis=0, keepdims=True)
            results = np.concatenate([results, tmp_results], 0)
            results_row.append("streaknet{}_{}_mean".format(n_version, n_type))
            
            tmp_results = np.zeros_like(tmp_results)
            results = np.concatenate([results, tmp_results], 0)
            results_row.append("")
    
    logger.info("Start collecting [noise bandpass]...")
    pbar = tqdm(total=len(network_version)*len(network_type)*len(exp_id)*len(noise_rate))
    for noise in noise_rate:
        for n_version in network_version:
            for n_type in network_type:
                for eid in exp_id:
                    path = os.path.join("StreakNet_outputs", "streaknet{}_{}_{}".format(n_version, n_type, eid), "noise_{}_bandpass_log.xlsx".format(noise))
                    df = pd.read_excel(path)
                    if results is None:
                        results = df.values 
                    else:
                        results = np.concatenate([results, df.values], 0)
                    results_row.append("bandpass_streaknet{}_{}_{}_noise{}".format(n_version, n_type, eid, noise))
                    pbar.update()
                
                tmp_results = results[-len(exp_id):]
                tmp_results = np.mean(tmp_results, axis=0, keepdims=True)
                results = np.concatenate([results, tmp_results], 0)
                results_row.append("bandpass_streaknet{}_{}_noise{}_mean".format(n_version, n_type, noise))
                
                tmp_results = np.zeros_like(tmp_results)
                results = np.concatenate([results, tmp_results], 0)
                results_row.append("")
    
    logger.info("Start collecting [noise reconstruction]...")
    pbar = tqdm(total=len(network_version)*len(network_type)*len(exp_id)*len(noise_rate))
    for noise in noise_rate:
        for n_version in network_version:
            for n_type in network_type:
                for eid in exp_id:
                    path = os.path.join("StreakNet_outputs", "streaknet{}_{}_{}".format(n_version, n_type, eid), "noise_{}_reconstruction_log.xlsx".format(noise))
                    df = pd.read_excel(path)
                    if results is None:
                        results = df.values 
                    else:
                        results = np.concatenate([results, df.values], 0)
                    results_row.append("streaknet{}_{}_{}_noise{}".format(n_version, n_type, eid, noise))
                    pbar.update()
                
                tmp_results = results[-len(exp_id):]
                tmp_results = np.mean(tmp_results, axis=0, keepdims=True)
                results = np.concatenate([results, tmp_results], 0)
                results_row.append("streaknet{}_{}_noise{}_mean".format(n_version, n_type, noise))
                
                tmp_results = np.zeros_like(tmp_results)
                results = np.concatenate([results, tmp_results], 0)
                results_row.append("")
    
    df = pd.DataFrame(results, index=results_row, columns=results_col)
    print(df)
    output_path = os.path.join("StreakNet_outputs", "summarized_results.xlsx")
    df.to_excel(output_path, engine='openpyxl')
    