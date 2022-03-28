# Stanford Dogs Breed Classifier

### Final Web Application :

![classifier](https://user-images.githubusercontent.com/61115039/160357524-991b375d-ce3a-4911-92ba-dc40665b0188.PNG)

## Web application Link of AZURE Cloud : https://dogclass.azurewebsites.net/

## Description
The <a href= "http://vision.stanford.edu/aditya86/ImageNetDogs/">Stanford Dogs Dataset</a> contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. It was originally collected for fine-grain image categorization, a challenging problem as certain dog breeds have near identical features or differ in colour and age.

I have used the InceptionV3 CNN Model, which is pre-trained on the ImageNet dataset for classification. Data augementation has been used for making the model generalize better and also to avoid overfitting. The model achieved an accuracy of 90% on validation set, which is decent for this dataset.I have also used the docker and Local git and deployed the model in AZURE Cloud.


### Pre-Requisites
For running the notebook on your local machine, following pre-requisites must be satisfied:
- NumPy
- Pandas
- Scikit-image
- IPython
- Matplotlib
- Tensorflow 2.X
- Keras
- AZURE Cloud

## The Dataset
The Stanford Dogs dataset has images of 120 dog breeds from around the world.

The contents of the dataset are:
- ***Number of Categories*** = 120
- ***Total Number of images*** = 20,580

The number of images per dog breed is low to train a neural network from scratch.
Hence it would be beneficial to use transfer learning. 
The dataset is imported from Kaggle.
The data can be found at https://www.kaggle.com/jessicali9530/stanford-dogs-dataset

More information about the dataset can be found at http://vision.stanford.edu/aditya86/ImageNetDogs/


### Installation
**Dependencies:**
```
# With Tensorflow CPU
git clone <repository name> 
pip install -r requirements.txt

# With Tensorflow GPU
pip install -r requirements-gpu.txt
```
**Nvidia Driver (For GPU, if you haven't set it up already):**
```
# Ubuntu 20.04
sudo apt-add-repository -r ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430

# Windows/Other
https://www.nvidia.com/Download/index.aspx
![example images](https://i.imgur.com/Mp2Te2Y.png)

```
## Approach
### Data Augmentation
Data augmentation is done through the following techniques:
- Rescaling (1./255)
- Shear Transformation (0.2)
- Zoom (0.2)
- Horizontal Flipping
- Rotation (20)
- Width Shifting (0.2)
- Height Shifting (0.2)

### Model Details
![Model Details](/images/model_details.PNG)

### Training Results
![Model Accuracy and Loss](/images/train_acc_loss.png)

### Final Video of Classifier

https://user-images.githubusercontent.com/61115039/160356753-4e43aeab-1ef0-480d-931b-8cf532e221c3.mp4



## References
- The original data source is found on http://vision.stanford.edu/aditya86/ImageNetDogs/ and contains additional information on the train/test splits and baseline results.
- Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.  <a href="http://people.csail.mit.edu/khosla/papers/fgvc2011.pdf">[pdf]</a> <a href="http://vision.stanford.edu/documents/KhoslaJayadevaprakashYaoFeiFei_FGVC2011.pdf">[poster]</a> <a href="http://vision.stanford.edu/bibTex/KhoslaJayadevaprakashYaoFeiFei_FGVC2011.bib">[BibTex]</a>
- J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009.  <a href="http://www.image-net.org/papers/imagenet_cvpr09.pdf">[pdf]</a> <a href="http://www.image-net.org/papers/imagenet_cvpr09.bib">[BibTex]</a>
- Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. Rethinking the Inception Architecture for Computer Vision, arXiv:1512.00567v3, 2015. <a href= "https://arxiv.org/pdf/1512.00567v3.pdf">[pdf]</a>

## :heart: Owner
Made with :heart:&nbsp;  by [Aman Chauhan](https://github.com/amanchauhan71)
