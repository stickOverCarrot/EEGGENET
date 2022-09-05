# EEG_GENet：A Feature-Level Graph Embedded Method for Motor Imagery Classification based on EEG Signal 
## Paper link: https://www.sciencedirect.com/science/article/pii/S0208521622000766?casa_token=_7Ii_umHfj0AAAAA:RzKOVOKyvHmfiG2AudWu4pyMd24Kiasgxyo34bBH7EXQHsNyoRkXORyy1iQdgFVWm3hraYIOhIE
## Environment
* python 3.7
* pytorch 1.8.0
* braindecode package is directly copied from https://github.com/robintibor/braindecode/tree/master/braindecode for preparing datasets 
## Start
* setp 1 Prepare dataset(Only needs to run once)
   
    `python tools/data_bciciv2a_tools.py --data_dir ~/dataset/bciciv2a/gdf -output_dir ~/dataset/bciciv2a/pkl`
* step 2 Train model 
  
    `python main_bciciv2a.py -data_dir ~/dataset/bciciv2a/pkl -id 1`
## Licence
For academtic and non-commercial usage
