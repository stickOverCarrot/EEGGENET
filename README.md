# EEGGENET
* setp 1 Prepare dataset(Only needs to run once) 
`python tools/data_bciciv2a_tools.py --data_dir ~/dataset/bciciv2a/gdf -output_dir ~/dataset/bciciv2a/pkl`
* step 2 Train model 
`python main_bciciv2a.py -data_dir ~/dataset/bciciv2a/pkl -id 1`
