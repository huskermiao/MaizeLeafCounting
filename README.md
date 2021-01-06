This Git Repo collects Python scripts used in the maize leaf counting paper

- **CountingByRegression_CNNs**: python scripts and example data for training and evaluating counting-by-regression CNN models.
- **CountingByDetection_FasterRCNNs**: python scripts and example data for training and evaluating counting-by-detection Faster-RCNN models
- **zooniverse_tools**: python scripts for uploading images to and downloading annotation results from Zooniverse.

In order to run these python scripts, you need to install the dependent packages first. You can use `pip` to create a running environment like this `pip install -r requirements.txt`. The code was tested under python 3.8.0 and you can find all dependent packages in `requirements.txt`.

Please contact Chenyong Miao (cmiao@huskers.unl.edu) if you have any questions when using these scripts. 

Citations:
- Miao, C., Guo, A., Yang, J., Ge, Y., & Schnable, J. C. (2020). Automation of Leaf Counting in Maize and Sorghum Using Deep Learning. bioRxiv. doi: https://doi.org/10.1101/2020.12.19.423626
- Miao, C., Hoban, T. P., Pages, A., Xu, Z., Rodene, E., Ubbens, J., ... & Schnable, J. C. (2019). Simulated plant images improve maize leaf counting accuracy. bioRxiv. doi: https://doi.org/10.1101/706994
