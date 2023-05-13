"""Script to visualize the log files from the baseline model, the model with pre-layer normalization, 
and the model with post-layer normalization. 

Authors: Zachary W. Hopton (22-737-027); Allison Ponce de Leon Diaz (22-740-633)

Example Call: mt-exercise-4 % python scripts/log_analysis.py --input-dir "models/deen_transformer_baseline" "models/deen_transformer_pre" "models/deen_transformer_post" --output-dir "models"
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser("Program to visualize machine translation model training progress")

    parser.add_argument("--input-dir", type = str, nargs="+", required=True,
                        help = "Directories containing the .log files to be analyzed")
    parser.add_argument("--output-dir", type = str,
                        help = "Directory to output log visualizations")
    
    return parser.parse_args()


def collect_ppl(dirs:list)->dict:
    perplexities = {}

    for dir in dirs:
        if "pre" in dir:
            id = "Prenorm"
        elif "post" in dir:
            id = "Postnorm"
        else:
            id = "Baseline"
        perplexities[id] = {}
        log = [x for x in os.listdir(dir) if x[-3:]=="log"].pop()

        step = 500
        with open(dir+"/"+log, "r", encoding="utf-8") as current:

            for line in current:
                if "joeynmt.prediction" in line and " ppl:" in line:
                    line = line.split(",")
                    ppl =line[2][6:]
                    perplexities[id][step]=float(ppl)
                    step+=500

    return perplexities


def make_outputs(out_dir, df):
    #first output table
    df.to_csv(path_or_buf = out_dir+ "/val_ppl_table.csv", index_label = "Validation ppl")

    #Now make graph
    viz = sns.lineplot(data = df)
    plt.savefig(out_dir+"/val_ppl_fig.png")


def main():
    args = get_args()
    
    ppl = collect_ppl(args.input_dir)

    ppl_df = pd.DataFrame(ppl)
    ppl_df.index.name = "Step"
    ppl_df.columns.name = "Validation ppl"
    #read the final thing to a csv for the table, then us sns and plt to save img of the three col
    make_outputs(args.output_dir, ppl_df)


if __name__ =="__main__":
    main()