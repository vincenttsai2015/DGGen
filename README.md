# A Deep Probabilistic Framework for Continuous Time Dynamic Graph Generation

This repository is the official implementation of *A Deep Probabilistic Framework for Continuous Time Dynamic Graph Generation*. Please note that this is an anonymized version of the implementation created for review purposes.

## Repository Structure

This repository consists of the following top-level directories

1. `config`: contains the configuration files for each dataset. Hyperparameter values in these files correspond to the final values used in our experiment. See Appendix C for details on values used for hyperparameter search.
2. `data`: Contains raw data for the novel Bikeshare dataset. Processed data for other datasets and Bikeshare will be populated automatically as needed.
3. `DGB`: As described in our main work, we use DGB [1] for link prediction evaluation. This directory should contain their [open source code](https://github.com/fpour/DGB) (in accordance to their license). In order to conform to space requirements for the supplementary material, we don't include the repository in our zip file and ask that you clone it into the empty directory before running the experiments (See setup below.)
4. `results`: Initially contains only baseline results. Automatically populated with results as described in detail in the following sections.
5. `saved_models`: Initially empty. Populates with the saved weights of model with best validation loss during training.
6. `DGGen`: Contains the main scripts for reproducing our experiments. Details are below.

## Setup

Install requirements:

```
pip install -r requirements.txt
```

Please note that installing Pytorch and Pytorch Geometric may require special install procedures depending on platform. Please refer to [Pytorch](https://pytorch.org/get-started/locally/) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) documentation for more details.


Install DGB [1]:

```
cd DGB
git clone https://github.com/fpour/DGB.git
```


## Data

All datasets used are freely available and used according to their respective licenses. The `data` directory contains raw data for the Bikeshare dataset. The datasets provided by Jodie [2] will automatically be downloaded
into this directly when using the training script with each Jodie dataset.

## Training

To train DG-Gen according to method described in the main paper, first modify the configuration file for the dataset you wish to use. The configuration files for all datasets are in the `config` top-level directory.
If no values are modified, the model will train according to the hyperparameters selected as part of our hyperparameter search process (described in Appendix C). You can then begin training by running the `train.py`
with the configuration file as the first command line argument. For example, for the Reddit dataset:

```
python DGGen/train.py config_reddit.json
```

## Generation
To generate synthetic interactions from a trained model, after running the `train.py` script, run `generate.py` with the dataset as the first command line argument. For example, after training DGGen as described above for
the Reddit dataset, we can then generate a synthetic CTDG as:

```
python DGGen/generate.py reddit
```

This will create two outputs forms with the CTDG: one as `.csv` and one as a pickled Pytorch Geometric Data object. Both files can be found in `results/synthetic_data`.


## Evaluation: Feature Properties
After training and generation for _all_ datasets is complete, feature properties can be calculated and visualized using:

```
python DGGen/features.py DGGen/eval.yaml
```

`eval.yaml` contains the paths to all relevant data for feature property calculation. These values will be correct by default if training and evaluation is conducted according to the instructions in the preceeding sections.

This will calculate the distances between source and synthetic graph features (analagous to Table 3 in our work) as well as produce histogram visualizations as in Figure 3 of our work. The Figure will be saved in the `results` top level directory.

## Evaluation: Topological Properties
Given that our topological property evaluation is identical to that of TIGGER-I [3], their (open source) evaluation script can be used on our generated synthetic graphs. Their open source respository is available [here](https://github.com/data-iitd/tigger). To do so, we first convert our generated synthetic data into a format compatible with their evaluation scripts. We can use the script `convert_for_baseline.py` and pass the dataset name as the first argument. For example:

```
python DGGen/convert_for_baseline.py reddit
```

This will create a TIGGER-I [3] compatible `.csv` file in `results/synthetic_data/baseline_compatible`.

As described in our main work, we next need to create discrete snapshots of our data and the source data (though for convenience, we have already done this for the source data) in order to compare topological properties. For example for the Reddit dataset:

```
python create_snapshots.py reddit
```

We note that `create_snapshots.py` is a modified version of a script from the TIGGER-I repository. All files adapted from the TIGGER-I repository are clearly marked as such at the top of the file.

Finally, we can calculate the topological metrics analagous to Table 2 and Table 10 in our work. For example for the Reddit dataset:

```
python topology.py reddit
```

## Evaluation: Link Prediction

Running the `train.py` script as described above will automatically run link prediction and calculate the relevant statistics that are reported in our work. These results will be saved as `.csv` files to the `results` top-level directory.


## Random seeding

As described in Appendix A of our work, all of our experiments are seeded insofar as possible and run multiple times to ensure reproducability. For these experiments all models are run 10 times with random seed starting at 100 for the first experiment
and increased by 1 for each subsequent eperiment (100-109).

## References
[1] Poursafaei, Farimah, et al. "Towards better evaluation for dynamic link prediction." Advances in Neural Information Processing Systems 35 (2022): 32928-32941.

[2] Kumar, Srijan, Xikun Zhang, and Jure Leskovec. "Predicting dynamic embedding trajectory in temporal interaction networks." Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 2019.

[3] Gupta, Shubham, et al. "Tigger: Scalable generative modelling for temporal interaction graphs." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 6. 2022.
