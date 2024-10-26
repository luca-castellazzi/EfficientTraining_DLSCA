# Towards Efficient Training in Deep Learning Side-Channel Attacks 

This repository stores the code used for the development of my Master's Thesis @ Politecnico di Milano (A.Y. 2021-2022), entitled _Towards Efficient Training in Deep Learning Side-Channel Attacks_.
The Thesis' goal is to study how multiple devices, several encryption-keys and different amount of train-data influence the performance of a Deep Learning based Side-Channel Attack (DL-based SCA) against AES-128.


## Repo Structure
### Folder notebooks_with_pdf
Contains two complete DL-based SCAs: the first targets the 6th byte of the key and the second shows how it is possible to recover the whole key.

### Folder src/scripts
Contains all runnable scripts used for the Thesis, which include key generation, Device-Key Trade-off Analysis (DKTA) and Device-Trace Trade-off Analysis (DTTA).

### Folder src/modeling
Contains Neural Network definition and Hyperparameter Tuning method, implemented with a Genetic Algorithm.

### Folder src/utils
Contains all utils functions used within the main scripts.


## Notes
The collected traces used in this project are available on Zenodo at [this link](https://doi.org/10.5281/zenodo.7817187). 


## License
[GPLv3](LICENSE.txt)
