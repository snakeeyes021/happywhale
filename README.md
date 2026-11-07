# ID the Seas: Whale and Dolphin Species classification using Convolutional Neural Networks

Is it possible to identify whale and dolphin species by looking only at dorsal fins? I've trained a machine learning model with 96% accuracy to do just that, and deployed it as a user-friendly [web app](https://huggingface.co/spaces/snakeeyes021/id-the-seas) for all to use.

![flavio-gasperini-gCpr7F7TU7s-unsplash](https://user-images.githubusercontent.com/26641674/157836489-40772f53-1e5a-4deb-b953-9de8bcc7f3db.jpg)

## Overview and Business Understanding
In the study of marine mammals, the ability to identify individuals or their pods is critical, allowing for population tracking over time and assessment of population trends and statistics. In this domain, there are multiple levels at which one could make meaningful contribution to groups or individuals in the field. 

At a more fundamental level, there's the task of species identification. While professional researchers would generally have no problem identifying a particular species at sight, whale watchers, citizen scientists, amateurs, and general marine enthusiasts would almost certainly benefit from quick and easy tools to aid in species ID.

For reasearchers and academics, the holy grail is individual identification. Currently, ID of individual whales and dolphins is performed manually by researchers, comparing multiple photographs side by side, checking for individual shapes and markings with just their own eyes. This is extraordinarily time consuming and not always perfectly accurate. Finding a way to do this faster would save immense amounts of time and resources that could be put to better use elsewhere.

To that end, Happywhale, a Washington state based research collaboration and citizen science platform, is seeking partners to develop machine learning models to aid in the process of whale and dolphin ID, with the hope to reduce ID times by over 99%. 

For now, the scope of this project includes only the first problem. However, upon completion, I will likely try my hand at the much more difficult and time-consuming second problem. Keep an eye out for additional branches.

## The Dataset
Happywhale has a dataset available as part of a [Kaggle competition](https://www.kaggle.com/c/happy-whale-and-dolphin). The dataset consists of over 50,000 images of whales and dolphins, categorized both by 25 different species and about 15,000 individuals. The images generally only contain dorsal fins and dorsal ridges.

For anomaly detection, I've also obtained a [miniature subset of the ImageNet dataset](https://www.kaggle.com/ifigotin/imagenetmini-1000), for an extra 35,000 images.

Here is a small sample of several species:

<p align="center">
  <b>Beluga Whale</b>
</p>
<p align="center">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839450-0889f045-b8f5-43eb-9b2a-13b24f9e5b9b.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839268-3c1cafa7-ec7c-489a-827d-cc5b7c14b4d6.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839286-bba22a06-2ced-484b-b8c8-130c4bc88a3d.jpg">
</p>

<p align="center">
  <b>Dusky Dolphin</b>
</p>
<p align="center">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839364-8b0c4e0b-e212-4b35-92ce-91dc2ad242c0.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839350-4fcc6f18-664f-4c10-a61f-35d0cc997068.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839381-e8a6467a-810e-4c5c-8cf0-10ac91decb93.jpg">
</p>

<p align="center">
  <b>Killer Whale</b>
</p>
<p align="center">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839507-e356953b-1e5f-428a-bfa7-de56ed8a4809.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839522-54fd98c8-f6c6-40c9-a852-2473609fede1.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839537-0fa190ae-e7ed-48f3-97bf-36c299086629.jpg">
</p>

<p align="center">
  <b>Southern Right Whale</b>
</p>
<p align="center">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839585-dbaa1d43-3c2a-43fc-92e4-be851411d9b1.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839557-83048bbb-123b-4f23-a87f-91631d96c505.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839576-75f6b9af-e97c-4d85-8805-f1fa15969766.jpg">
</p>

<p align="center">
  <b>Commerson's Dolphin</b>
</p>
<p align="center">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839623-fb687604-b16e-48a8-bceb-85f1d09de4a1.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839642-9305ddd2-0197-45c2-b889-7f623ad030bd.jpg">
  <img width="250" src="https://user-images.githubusercontent.com/26641674/157839646-b272320c-79e1-4f1f-a9f6-61ab0eb0be22.jpg">
</p>

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
There are two independent models in the final web app, a species classifier and an anomaly detector. 

For the species classifier, I began with a basic untuned convolutional neural network as a baseline. I had begun by training on a TensorFlow ImageDataGenerator object, but for the size of the dataset, this consumed immense amounts of time to train, so I let the model train for a single epoch (about 12 hours) and called that my baseline. I switched to using image_dataset_from_directory, which sped up the training time by multiple orders of magnitude. For my subsequent models, I experimented with image input size and color mode, and I also experimented with different overfitting avoidance strategies (L2 regularization, dropout, data augmentation, etc.). My final model was a DenseNet121. I began by chopping off the top layer and adding a few of my own, loading the pre-trained ImageNet weights, and freezing all but my final layers. I then trained and gradually unfroze portions of the model at a time. In the end, this yielded very poor results, so I took the exact opposite approach and trained from scratch with nothing frozen. This was better, but there was still room for improvement. Finally, I loaded the pre-trained weights but froze nothing. This yielded the best results, so I kept it as my final species classifier model. 

For the anomaly detector, running short on GPU resources for additional training, I wanted the quickest and simplest rule-based method of anomaly detection that I could come up with, based on the work I had already done. So I took my species classifier model and replaced the final softmax activation with a linear activation to get the log-odds for the different class predictions. Examining the distribution of the log-odds for the Happywhale training dataset as well as the log-odds distribution for the mini-ImageNet dataset, I calculated the mean and standard deviation of each distribution. Then I merely compared the z-scores of a given image against each distribution, and, if the Happywhale z-score for that image was higher than its ImageNet z-score, it was flagged as an anomaly. This yielded very decent results on unseen data (91% accuracy), so I kept it as my anomaly detector model.

## Evaluation
### Best Species Classifier
![Model Results](https://user-images.githubusercontent.com/26641674/157840178-da41c7b6-1f6b-4685-be3e-0c58dae4b2a8.png)

The final species classifier achieved an accuracy of 96% up from the initial baseline of 54%. The final classifier performs better with species that have distinctive appearances, but it generally does ok with species with less distinctive appearances *as long as they're decently well represented in the training data.* 

Not specific to species there are some further limitations to the classifier. Some of the things that give it trouble are body parts other than dorsal fins, image corruption, badly injured fins, too many additional elements in the frame, subjects appearing too small in the frame, difficult angles, difficult cropping, or multiple individuals in the image. These sorts of issues arise in a significant number of the 4% of incorrectly classified test data. 

### The anomaly detector 
As pointed out above, the anomaly detector has an accuracy of 91%. Here is its confusion matrix:
<p align="left">
  <img width="500" src="https://user-images.githubusercontent.com/26641674/158005524-3927bc33-7a70-4d42-b07e-9c4a70cfdbe9.png">
</p>

## Deployment and Next Steps
I've deployed the two models as a simple and user-friendly [web app](https://huggingface.co/spaces/snakeeyes021/id-the-seas). Since both models are technically separate, an image that is flagged as an anomaly will still receive a class prediction. This is desired behavior as we would still want a class prediction for false positives from the anomaly detector. The app works on smartphone or computer. Either platform can upload images, but use on a smartphone allows for capturing images with the phone's camera. 
![happywhale app](https://user-images.githubusercontent.com/26641674/157893260-39cd6477-610b-4190-95f5-7132915c120d.png)

Moving forward, building on the work I've done for species identification and applying new techniques and models, such as are common to facial recognition, I'm planning on tackling the individual ID problem. On said problem, my best DenseNet performs 100x better than random guessing, which amounts to an accuracy of a whole 1%, so there's certainly room for improvement! Keep an eye out for additional branches in this repo to that end. 

## Repository Structure
**On Reproducibility:** The Conda environment necessary to run this project as is can be found in the main directory of this repo as environment.yml.

**On Models and Data:** Models and data for this project are too large to be stored on GitHub. The dataset can be downloaded [here](https://www.kaggle.com/c/happy-whale-and-dolphin/data) and should be unzipped into a folder called 'data' in the same directory as this notebook. A second dataset for anomaly detection can be found [here](https://www.kaggle.com/ifigotin/imagenetmini-1000) and should be unzipped into 'data/not-whales-or-dolphins' and the folders containing killer whales and grey whales should be deleted (folders n02066245 and n02072394). 

Models can be downloaded [here](https://drive.google.com/drive/folders/1LQjT3ViklSZU469KWLJCW65KBXlwgb6G?usp=sharing) and should be stored in a folder called 'models' in the same directory as this notebook.

```


├── misc_notebooks
|   ├── Final Notebook-Full Training.ipynb
|   ├── anomaly-detection.ipynb
|   ├── directory-method.ipynb
|   ├── image-previews.ipynb
|   ├── individual_id.ipynb
|   ├── initial-setup-bak.ipynb
|   ├── initial-setup-final.ipynb
|   ├── initial-setup.ipynb
|   ├── simple-model-train.ipynb
|   ├── transfer-learning.ipynb

├── .gitignore
├── Final Notebook.ipynb
├── Presentation.pdf
├── README.md
└── environment.yml
```
## For more information
Check out the full [Jupyter notebook](https://github.com/snakeeyes021/happywhale/blob/main/Final%20Notebook.ipynb) and the [presentation](https://raw.githubusercontent.com/snakeeyes021/happywhale/main/Presentation.pdf).
