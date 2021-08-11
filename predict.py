# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:42:36 2019

@author: SY
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
# from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.parsing import parse_predict_args, modify_predict_args
from chemprop.train import make_predictions
from chemprop.data.utils import get_task_names

if __name__ == '__main__':
    args = parse_predict_args()
    modify_predict_args(args)
    task_names = get_task_names(path=args.test_path,
                                target_columns=args.target_columns)
    # pred, smiles = make_predictions(args, df.smiles.tolist())
    # df = pd.DataFrame({'smiles': smiles})
    ##################my addition###################
    pred, smiles = make_predictions(args)
    if args.smiles_columns is not None:
        df = pd.DataFrame({args.smiles_columns: smiles})
    else:
        df = pd.DataFrame({'smiles': smiles})
    ################################################
    for i in range(len(pred[0])):
        df[task_names[i]] = [item[i] for item in pred]
    df.to_csv(args.preds_path, index=False)