# ID the Seas: Whale and Dolphin Species classification using Convolutional Neural Networks

Is it possible to identify whale and dolphin species by looking only at dorsal fins? I've trained a machine learning model with 96% accuracy to do just that, and deployed it as a user-friendly [web app](https://huggingface.co/spaces/snakeeyes021/id-the-seas) for all to use.

![flavio-gasperini-gCpr7F7TU7s-unsplash](https://user-images.githubusercontent.com/26641674/157836489-40772f53-1e5a-4deb-b953-9de8bcc7f3db.jpg)


## Overview and Business Understanding
In the study of marine mammals, the ability to identify individuals or their pods is critical, allowing for population tracking over time and assessment of population trends and statistics. In this domain, there are multiple levels at which one could make meaningful contribution to groups or individuals in the field. 

At a more fundamental level, there's the task of species identification. While professional researchers would generally have no problem identifying a particular species at sight, whale watchers, citizen scientists, amateurs, and general marine enthusiasts would almost certainly benefit from quick and easy tools to aid in species ID.

For reasearchers and academics, the holy grail is individual identification. Currently, ID of individual whales and dolphins is performed manually by researchers, comparing multiple photographs side by side, checking for individual shapes and markings with just their own eyes. This is extraordinarily time consuming and not always perfectly accurate. 

Happywhale, a Washington state based research collaboration and citizen science platform, is seeking partners to develop machine learning models to aid in the process of whale and dolphin ID, with the hope to reduce ID times by over 99%. 

## The Dataset
Happywhale has a dataset available as part of a [Kaggle competition](https://www.kaggle.com/c/happy-whale-and-dolphin). The dataset consists of over 50,000 images of whales and dolphins, categorized both by 25 different species and about 15,000 individuals. 

For anomaly detection, I've also obtained a [miniature subset of the ImageNet dataset](https://www.kaggle.com/ifigotin/imagenetmini-1000), for an extra 35,000 images.

Here is a small sample of several species:

![0c2e779db3d141](https://user-images.githubusercontent.com/26641674/157839268-3c1cafa7-ec7c-489a-827d-cc5b7c14b4d6.jpg)
![2ebc331bc957b7](https://user-images.githubusercontent.com/26641674/157839450-0889f045-b8f5-43eb-9b2a-13b24f9e5b9b.jpg)
![2dd94b169ea511](https://user-images.githubusercontent.com/26641674/157839286-bba22a06-2ced-484b-b8c8-130c4bc88a3d.jpg)

![0e9ed770b071d5](https://user-images.githubusercontent.com/26641674/157839350-4fcc6f18-664f-4c10-a61f-35d0cc997068.jpg)
![0cfcde0ab18d1f](https://user-images.githubusercontent.com/26641674/157839364-8b0c4e0b-e212-4b35-92ce-91dc2ad242c0.jpg)
![1baa7257902c8d](https://user-images.githubusercontent.com/26641674/157839381-e8a6467a-810e-4c5c-8cf0-10ac91decb93.jpg)

![0f69d8bec6e33a](https://user-images.githubusercontent.com/26641674/157839507-e356953b-1e5f-428a-bfa7-de56ed8a4809.jpg)
![0a4de2e78b94be](https://user-images.githubusercontent.com/26641674/157839522-54fd98c8-f6c6-40c9-a852-2473609fede1.jpg)
![0f58b6a8ff5ed7](https://user-images.githubusercontent.com/26641674/157839537-0fa190ae-e7ed-48f3-97bf-36c299086629.jpg)

![3dace1d4074b97](https://user-images.githubusercontent.com/26641674/157839557-83048bbb-123b-4f23-a87f-91631d96c505.jpg)
![2bfda90ac3c436](https://user-images.githubusercontent.com/26641674/157839585-dbaa1d43-3c2a-43fc-92e4-be851411d9b1.jpg)
![0b6099fb23c990](https://user-images.githubusercontent.com/26641674/157839576-75f6b9af-e97c-4d85-8805-f1fa15969766.jpg)

![1c55d7d9af126e](https://user-images.githubusercontent.com/26641674/157839623-fb687604-b16e-48a8-bceb-85f1d09de4a1.jpg)
![a1148c11e1f880](https://user-images.githubusercontent.com/26641674/157839642-9305ddd2-0197-45c2-b889-7f623ad030bd.jpg)
![e713b9f8239a9d](https://user-images.githubusercontent.com/26641674/157839646-b272320c-79e1-4f1f-a9f6-61ab0eb0be22.jpg)

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
![Model Results](https://user-images.githubusercontent.com/26641674/157840178-da41c7b6-1f6b-4685-be3e-0c58dae4b2a8.png)


## Deployment and Next Steps


## Repository Structure
```


├── data


├── .gitignore

├── Presentation.pdf

├── Final Notebook.ipynb

├── README.md
```
## For more information
Check out the full [Jupyter notebook]() and the [presentation]().
