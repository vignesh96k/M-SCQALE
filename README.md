# <p align="center">Quality Assessment of Low-light Restored Images: A Subjective Study and an Unsupervised Model</p>
<p align="center">
Vignesh Kannan, Sameer Malik, Nithin Babu and Rajiv Soundararajan
</p>

<p align="center">
<a> DSLR Dataset and Official Pytorch Code of the IEEE Access 2023 paper:</a><br>

<p align="center">
<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10172189">Quality Assessment of Low-light Restored Images: A Subjective Study and an Unsupervised Model</a>

<p align="center">
<a href="http://ece.iisc.ac.in/~rajivs/databases/DSLR.zip">Link to DSLR Dataset</a>
  
<p align="center">
<a href="https://docs.google.com/forms/d/e/1FAIpQLSfMO2dSZPTldyyHKoa5I4fYKleR5WtaHYYVukip-9NtKpi8OA/viewform?usp=sf_link">Form to receive password to Dataset ZIP file</a>


</p>

![Architecture](./M-SCQALE_block.JPG)
## Environment
Our code has been tested with the following env specs:
- Python **3.6.13**
- CUDA **11.3.1**
- Pytorch **1.10.1**
- Torchvision **0.11.2**



### Setting up Conda environment
Execute the following lines for setting up the conda environment
```
conda create env -f M-SCQALE.yml
conda activate M-SCQALE
```

## Training

To train the model, run the following:
```
bash runtrain.sh
```

## Performance evaluation
### M-SCQALE Pre-trained weights
Google Drive link for pre-trained weights:
- M-SCQALE [link](https://drive.google.com/file/d/1gn2beZEcI67FgcXg_I6wYNAsqLw3qOmx/view?usp=sharing)

Extract the downloaded zip file in base folder.


### Setting up pristine patches
Link for pre-selected pristine patches [link](https://drive.google.com/file/d/1aZYVpSn4Z_b_74J8_37sCRqQshDHhtFd/view?usp=drive_link).
Copy the downloaded file to the base folder
## Testing 
Testing code for evaluating the M-SCQALE model.
```
python EVAL.py --loadpatches
```

## Citation
If you find this work useful for your research, please cite our paper:
```
@ARTICLE{10172189,
  author={Kannan, Vignesh and Malik, Sameer and Babu, Nithin C. and Soundararajan, Rajiv},
  journal={IEEE Access}, 
  title={Quality Assessment of Low-Light Restored Images: A Subjective Study and an Unsupervised Model}, 
  year={2023},
  volume={11},
  number={},
  pages={68216-68230},
  doi={10.1109/ACCESS.2023.3292114}}
