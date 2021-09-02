import numpy as np
import pandas as pd
import pickle
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", "-i",
                        help="input file path",type=str)
    parser.add_argument("--output", "-o",
                        help="output file path",type=str)

    return parser.parse_args()

def Atom2MolFeature():
    args = parse_args()
    args.input = '/mnt/home/linjie/projects/CMPNN-master-copy/aevs_descriptor/result/train_aevs.pkl'
    args.output = '/mnt/home/linjie/projects/CMPNN-master-copy/aevs_descriptor/result/train_features.pkl'
    features_df = pd.read_pickle(args.input)['features']
    features = features_df.apply(lambda x: np.sum(x,axis=0))
    features.to_pickle(args.output)

def readPickl():
    args = parse_args()
    features_df = pd.read_pickle(args.output)
    output = np.array(features_df.apply(lambda x: x.tolist()).tolist())






if __name__ == "__main__":
    Atom2MolFeature()
    # readPickl()