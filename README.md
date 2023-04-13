# CMPNN_AEV
The code was built based on [DMPNN](https://github.com/chemprop/chemprop). Thanks a lot for their code sharing!

## Dependencies
+ cuda >= 8.0
+ cuDNN
+ RDKit
+ torch >= 1.2.0

## Usage
### 1.Download Data
First, you need to go to http://www.pdbbind.org.cn/download.php to download the PDBBIND Refined set v2016 and CASF v2016 in order to train/test the model.
### 2.Generate atomic environment vector
```
python  aevs_descriptor/train_aev_calculate_protein.py -train pdbbind2016/pdbbind_v2016_refined/refined_2016_pathdata.csv 
-d pdbbind2016/pdbbind_v2016_refined/refined-set -r 3.5 -b 256 -cm '{"X":["H","C","O","N","P","S","F","Cl","Br","I"]}' 
--removeHs -o aevs_descriptor/result3.5/train_aevs.pkl 
```
```
python aevs_descriptor/test_aev_calculate_protein.py -test pdbbind2016/CASF-2016/CoreSetPath.csv  
-d pdbbind2016/pdbbind_v2016_refined/refined-set -r 3.5 -b 256 -b 256 -e aevs_descriptor/result3.5/aevc.pth
-am aevs_descriptor/result3.5/amap.json -cm aevs_descriptor/result3.5/cmap.json
--removeHs -o aevs_descriptor/result3.5/train_aevs.pkl 
```
### 3.train CMPNN model
```
python train.py --data_path pdbbind2016/pdbbind_v2016_refined/refined_2016_pathdata.csv --separate_test_path 
pdbbind2016/CASF-2016/CoreSetPath.csv --separate_test_atom_descriptors_path aevs_descriptor/result3.5/test_aevs.pkl 
--atom_descriptors_path aevs_descriptor/result3.5/train_aevs.pkl --datapaths pdbbind_v2016_refined/refined-set --dataset_type regression 
--epochs 300 --num_folds 1 --gpu 0 --save_dir  ckpt/ckpt_debug3.5 --dropout 0.25 --batch_size 64 --smiles_columns smiles --target_columns affinity 
--ligand_file_type sdf --metric r2 --hidden_size 512 --overwrite_default_atom_features
```
### 4.predict
```
python predict.py --test_path pdbbind2016/CASF-2016/CoreSetPath.csv --atom_descriptors_path aevs_descriptor/result3.5/test_aevs.pkl 
--datapaths pdbbind2016/pdbbind_v2016_refined/refined-set --checkpoint_dir ckpt/ckpt_debug3.5 --preds_path ckpt/ckpt_debug3.5/predict.csv 
--smiles_columns smiles --target_columns affinity --ligand_file_type sdf
```
