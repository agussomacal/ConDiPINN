# Deep Learning-based Schemes for Singularly Perturbed Convection-Diffusion Problems

This repository has all the code implementation of the project for the proceedings of the CEMRACS 2021: 
[Deep Learning-based Schemes for Singularly Perturbed Convection-Diffusion Problems](https://github.com/agussomacal/ConDiPINN)

### Abstract

The experiments that can be found in this repository show different ways of approximating a 1 dimensional convection-diffusion equation 
$$
\varepsilon u''(x) + u'(x) = 1 \quad \forall x\in (0, 1) \\
10^{-3} u'(0) +  u(0) = 0 \\
10^{-3} u'(1) + u(1) = 0
$$
by using different variations of Physics-Informed Neural Networks (PINN) together with a comparison against the classical Finite Element Method (FEM).

### Examples

Three sets of experiments can be found in the scripts and in the jupyter-notebook:

* Varying number of training examples (10, 100, 1000): src/*Proceedings/n_train.py*
* Varying the sampling method (linspace, random): *src/Proceedings/sampling_method.py*
* Varying the machine precision (16, 32, 64): *src/Proceedings/float_precision.py*

If using binder, the jupyter-notebook will automatically load the experiments and the widgets will allow the user to choose different combinations of parameters to analyze with more flexibility the results of the paper. If other combinations would like to be explored, the above mentioned scripts should be modified.

### Setup for developers

Create virtual enviroment. python 3.6 is needed to be able to run tensorflow 1.13.1
```
python3.6 -m venv venv
```

Activate virtual enviroment
```
. .venv/bin/activate
```
Install libraries 
```
pip install -r requirements.txt 
```
Add the repository scripts path to the environment python path so it can call and import them correctly. For that, once the environment is created create a file with extension **.pth** in the virtual environment directory **env/lib/python3.6/site-packages/**, for example create the file **env/lib/python3.6/site-packages/path2self_packages.pth** and write in it two lines that tell python which are the paths to the project and the src folder, ex:

```
/home/user/Repositories/condipinn
/home/user/Repositories/condipinn/src
```

# Run experiments

### From shell



Change directory to **/condipinn/src/experiments** and run the three experiments:

```
python Proceedings/n_train.py
python Proceedings/float_precision.py
python Proceedings/sampling_method.py
```
To change the global parameters like *number_of_cores*, $\kappa$, $\alpha$, $\varepsilon$, number of repetitions, etc are in Proceedings/parameters.py.

### Jupyter notebooks

To run locally in the computer the jupyter notebooks with the set environment first install ipykernel and set the environment to run in jupyter with the following commands:
```
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "ConDiPINN-venv"
```
Source: https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments 

Then, all the experiments and plots of the proceedings with more flexible options to explore different combinations of parameters can be found in **src/notebooks/ProceedingsExperiments.ipynb**

   