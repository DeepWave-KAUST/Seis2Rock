![LOGO](https://github.com/DIG-Kaust/Seis2Rock/blob/main/assets/Seis2Rock_Banner_Repo.gif)

# Seis2Rock

> **<span style='color: blue;'>Seis2Rock: A Data-Driven Approach to Direct Petrophysical Inversion of Pre-Stack Seismic Data</span>** \
> Corrales M.<sup>1</sup>, Hoteit H.<sup>1</sup>, Ravasi M.<sup>1</sup>\
> <sup>1</sup> King Abdullah University of Science and Technology (KAUST)


## Project structure
This repository is organized as follows:

* :open_file_folder: **assets**: images and figures of the project.
* :open_file_folder: **seis2rock**: python library containing routines for seis2rock.
* :open_file_folder: **data**: folder containing data (and instructions on how to retrieve the data).
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details).


## Notebooks for Synthetic dataset
The following notebooks are provided:

- :orange_book: ``01_Synthetic_Benchmark.ipynb``: The notebook assesses and benchmarks the method's applicability using a synthetic dataset that has been constructed based on the reservoir model of the Smeaheia Field.
- :orange_book: ``02_Synthetic_4D_Benchmark.ipynb``: The notebook aims to evaluate the feasibility of tracking changes in water saturation using well logs that lack information about the new water-oil contact in the subsurface. It seeks to simulate 4D changes in the subsurface.
- :orange_book: ``03_Synthetic_Stacking_Wells.ipynb``: This notebook compares the inversion results obtained by varying the number of well logs used for training.

## Notebooks for Volve dataset
The following notebooks are provided:

- :blue_book: ``01_Wavelet_Estimation_Well_NO_15_9_19_BT2.ipynb``: This notebook demonstrates the procedure for extracting a statistical wavelet estimate along the Well NO 15-9 19 BT2 fence. This wavelet is subsequently utilized in the inversion step.
- :blue_book: ``02_Wavelet_Estimation_Well_NO_15_9_19_A.ipynb``: This notebook demonstrates the procedure for extracting a statistical wavelet estimate along the Well NO 15-9 19 A fence. This wavelet is subsequently utilized in the inversion step.
- :blue_book: ``03_Benchmark_Inversion_Synthetic_Well_NO_15_9_19_BT2.ipynb``: This notebook conducts a benchmark of the method using a synthetic gather constructed from the well log information of well NO 15-9 19 BT2.
- :blue_book: ``04_Benchmark_Inversion_Synthetic_Well_NO_15_9_19_A.ipynb``: This notebook conducts a benchmark of the method using a synthetic gather constructed from the well log information of well NO 15-9 19 A.
- :blue_book: ``05_Inversion_Well_NO_15_9_19_BT2.ipynb``: This notebook presents the inversion results obtained just on the well 15-9 19 BT2..
- :blue_book: ``06_Inversion_Well_NO_15_9_19_A.ipynb``: This notebook presents the inversion results obtained just on well NO 15-9 19 A.
- :blue_book: ``07_Inversion_Fence_Well_NO_15_9_19_BT2.ipynb``: This notebook presents the inversion results obtained along the 2D fence of well NO 15-9 19 BT2.
- :blue_book: ``08_Inversion_Fence_Well_NO_15_9_19_A.ipynb``: This notebook presents the inversion results obtained along the 2D fence of well NO 15-9 19 A.
- :blue_book: ``09_Inversion_Fence_Well_NO_15_9_19_BT2_stacking_wells.ipynb``: This notebook presents the inversion results obtained along the 2D fence using the well logs from NO 15-9 19 BT2 and NO 15-9 19 A for the training.
- :blue_book: ``10_Inversion_Fence_Well_NO_15_9_19_A_stacking_wells.ipynb``: This notebook presents the inversion results obtained along the 2D fence using the well logs from NO 15-9 19 BT2 and NO 15-9 19 A for the training.
- :blue_book: ``11_Inversion_3D.ipynb``: This notebook presents the inversion in 3D. 

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Αctivate the environment by typing:
```
conda activate seis2rock
```

After that you can simply install your package: (double check your new environment is active to proceed as follows)
```
pip install .
```
or in developer mode:
```
pip install -e .
```


> **Note** <br>
> All experiments have been carried on a Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz equipped with a single NVIDIA Quadro
> RTX 4000 GPU. Different environment configurations may be required for different combinations of workstation and GPU.


