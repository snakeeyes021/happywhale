# ID the Seas: Whale and Dolphin Species classification using Convolutional Neural Networks

Is it possible to identify whale and dolphin species by looking only at dorsal fins? I've trained a machine learning model with 96% accuracy to do just that, and deployed it as a user-friendly [web app](https://huggingface.co/spaces/snakeeyes021/id-the-seas) for all to use.

![image]()

## Overview and Business Understanding
In the study of marine mammals, the ability to identify individuals or their pods is critical, allowing for population tracking over time and assessment of population trends and statistics. In this domain, there are multiple levels at which one could make meaningful contribution to groups or individuals in the field. 

At a more fundamental level, there's the task of species identification. While professional researchers would generally have no problem identifying a particular species at sight, whale watchers, citizen scientists, amateurs, and general marine enthusiasts would almost certainly benefit from quick and easy tools to aid in species ID.

For reasearchers and academics, the holy grail is individual identification. Currently, ID of individual whales and dolphins is performed manually by researchers, comparing multiple photographs side by side, checking for individual shapes and markings with just their own eyes. This is extraordinarily time consuming and not always perfectly accurate. 

Happywhale, a Washington state based research collaboration and citizen science platform, is seeking partners to develop machine learning models to aid in the process of whale and dolphin ID, with the hope to reduce ID times by over 99%. 

## The Dataset
Happywhale has a dataset available as part of a [Kaggle competition](https://www.kaggle.com/c/happy-whale-and-dolphin). The dataset consists of over 50,000 images of whales and dolphins, categorized both by 25 different species and about 15,000 individuals. 

For anomaly detection, I've also obtained a [miniature subset of the ImageNet dataset](https://www.kaggle.com/ifigotin/imagenetmini-1000), for an extra 35,000 images.

Here is a small sample of several species:


![0c2e779db3d141](https://user-images.githubusercontent.com/26641674/157835621-1dfab97b-770b-4504-bfd0-d0ab6d7c6e91.jpg)




The full list of species is as follows:
Beluga Whale,
Blue Whale,
Bottlenose Dolphin,
Bryde’s Whale,
Commerson’s Dolphin,
Common Dolphin,
Cuvier’s Beaked Whale,
Dusky Dolphin,
False Killer Whale,
Fin Whale,
Frasier’s Dolphin,
Gray Whale,
Humpback Whale,
Killer Whale,
Melon-Headed Whale,
Minke Whale,
Pantropical Spotted Dolphin,
Pilot Whale,
Pygmy Killer Whale,
Rough-Toothed Dolphin,
Sei Whale,
Southern Right Whale,
Spinner Dolphin,
Spotted Dolphin,
White-Sided Dolphin


## Modeling


## Evaluation


## Deployment and Next Steps


## Repository Structure
```


├── data


├── .gitignore

├── Presentation.pdf

├── Combined Notebook.ipynb

├── README.md
```
## For more information
Check out the full [Jupyter notebook]() and the [presentation]().
