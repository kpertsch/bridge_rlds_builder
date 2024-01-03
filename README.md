# BridgeData RLDS Conversion

This repo converts data in the raw BridgeData format (i.e data saved by [bridge_data_robot](https://github.com/rail-berkeley/bridge_data_robot)) into RLDS format for [OXE](https://robotics-transformer-x.github.io/) integration.

## Installation

```
conda create -n bridge_rlds python=3.10
conda activate bridge_rlds 
pip install -r requirements.txt
```

## Convert Data
First, make sure that no GPUs are used during data processing with `export CUDA_VISIBLE_DEVICES=`. Then run:
```
cd bridge_dataset
tfds build --overwrite
```

The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/bridge_dataset`. You can specify a different output directory with the `--data_dir` flag. 

### Parallelizing Data Processing
By default, dataset conversion uses 10 parallel workers. If you are parsing a large dataset, you can increase the 
number of used workers by increasing `N_WORKERS` in the dataset class. Try to use slightly fewer workers than the 
number of cores in your machine (run `htop` in your command line if you don't know how many cores your machine has). 

The dataset value `MAX_PATHS_IN_MEMORY` controls how many filepaths will be processed in parallel before they get 
written to disk sequentially. As a rule of thumb, setting this value as high as possible will make dataset conversion
faster, but don't set it too high to not overflow the memory of your machine. Setting it to >10-20x the number of workers
is usually a good default. You can monitor `htop` during conversion and reduce the value in case your memory overflows.

### Visualize Converted Dataset
To verify that the data is converted correctly, please run the data visualization script from the base directory:
```
python visualize_dataset.py <name_of_your_dataset>
``` 
This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.
Note, if you are running on a headless server you can modify `WANDB_ENTITY` at the top of `visualize_dataset.py` and 
add your own WandB entity -- then the script will log all visualizations to WandB. 