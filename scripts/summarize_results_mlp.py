import os
import pandas as pd 
import numpy as np
from tqdm import tqdm
from loguru import logger


network_type = ['128', '256', '512']
exp_id = ['1', '2', '3', '4', '5']

if __name__ == "__main__":
    results = None
    results_col = ["acc", "prec", "recall", "f1", "psnr", "snr", "cnr", "var"] * 5
    results_row = []
    
    # bandpass mlp
    logger.info("Start collecting [bandpass mlp]...")
    pbar = tqdm(total=len(network_type)*len(exp_id))
    for n_type in network_type:
        for eid in exp_id:
            path = os.path.join("StreakNet_outputs", "mlp_{}_{}".format(n_type, eid), "bandpass_log.xlsx")
            df = pd.read_excel(path)
            if results is None:
                results = df.values 
            else:
                results = np.concatenate([results, df.values], 0)
            results_row.append("bandpass_mlp_{}_{}".format(n_type, eid))
            pbar.update()
        
        tmp_results = results[-len(exp_id):]
        tmp_results = np.mean(tmp_results, axis=0, keepdims=True)
        results = np.concatenate([results, tmp_results], 0)
        results_row.append("bandpass_mlp_{}_mean".format(n_type))
        
        tmp_results = np.zeros_like(tmp_results)
        results = np.concatenate([results, tmp_results], 0)
        results_row.append("")
    
    # reconstruction mlp
    logger.info("Start collecting [reconstruction mlp]...")
    pbar = tqdm(total=len(network_type)*len(exp_id))
    for n_type in network_type:
        for eid in exp_id:
            path = os.path.join("StreakNet_outputs", "mlp_{}_{}".format(n_type, eid), "reconstruction_log.xlsx")
            df = pd.read_excel(path)
            if results is None:
                results = df.values 
            else:
                results = np.concatenate([results, df.values], 0)
            results_row.append("mlp_{}_{}".format(n_type, eid))
            pbar.update()
        
        tmp_results = results[-len(exp_id):]
        tmp_results = np.mean(tmp_results, axis=0, keepdims=True)
        results = np.concatenate([results, tmp_results], 0)
        results_row.append("mlp_{}_mean".format(n_type))
        
        tmp_results = np.zeros_like(tmp_results)
        results = np.concatenate([results, tmp_results], 0)
        results_row.append("")
    
    df = pd.DataFrame(results, index=results_row, columns=results_col)
    print(df)
    output_path = os.path.join("StreakNet_outputs", "summarized_results_mlp.xlsx")
    df.to_excel(output_path, engine='openpyxl')
    