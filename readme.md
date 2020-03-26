# Deep Interactive Segmentation

Official repository for the two papers [**Getting to 99% Accuracy in Interactive Segmentation
**](https://arxiv.org/abs/2003.07932), submitted to Signal Processing: Image Communication the Special Issue on Computational Image Editing. 
and  
Interactive Training and Architecture for Deep Object Selection, accepted as oral presentation for ICME 2020.  
Marco Forte<sup>1</sup>, [François Pitié](https://francois.pitie.net/)<sup>1</sup>  

<sup>1</sup> Trinity College Dublin


## Requirements
GPU memory >= 4GB for inference on Berkeley and GrabCut. Optimal performance around 480p resolution.

#### Packages:
- torch >= 1.4
- numpy
- opencv-python
- [guided_filter_pytorch](https://pypi.org/project/guided-filter-pytorch/)
#### Additional Packages for jupyter notebook
- matplotlib


## Models
| Model Name  |     File Size   | NoC Grabcut  | NoC Berkeley |
| :------------- |------------:| :-----|----:|
| [SyntheticPretrained+Finetune on SBD](https://drive.google.com/file/d/1nJMTXSlprm5FQaQA5gfyU8CbSEX8ghzJ/view?usp=sharing)  | 144mb | 1.74 | 2.93  |
We will release more models shortly.


## Prediction 
We provide a script `demo.py` which evaluates our model in terms of mean IoU and number of clicks to reach 90% accuracy. Links to download: the [GrabCut](https://drive.google.com/open?id=1FFBH4vArby8alggT0SKjXPW7F8ShjXTp) and [Berkeley](https://drive.google.com/open?id=1atKWE4IY4FKFaNHsn-l7kbEo8T2z3MPx) datasets. Results are slightly improved from Table. 8 in the paper, this is due to changes in prediction, the weights are the same as used in the paper.


## Training
Training code is not released at this time. It may be released upon acceptance of the paper.

## Citation

```
@misc{forte2020InterSeg,
    title={Getting to 99% Accuracy in Interactive Segmentation},
    author={Marco Forte and Brian Price and Scott Cohen and Ning Xu and François Pitié},
    year={2020},
    eprint={2003.07932},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
### Related works of ours
 - [F, B, Alpha Matting](https://github.com/MarcoForte/FBA-Matting)
