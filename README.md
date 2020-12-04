# Is it Korean? or Indian? 
## Cuisine recognition from images using Neural Network

Food identification from images has gained interest from Machine Learning communities. With a proper recognition, food identification from images can be useful in understanding food composition, nutritional information, or understanding a person's dietary habits. Previously, high performance models has been established for identifying a single food item (i.e. type of fruits/vegetables/individual dishes).  The goal of this project is to use neural network to identify cuisine from food images. 

Cuisine identification can be helpful for i) a user who see's a photo of a dish and wants to identify which cuisine it is ii) for building auto tag generation for search engines or restaurant review apps for their data analysis. 

Additionally, the pipeline for this cuision recognizer can be extended to other object identifications. 

## The data
### Data source
- The data was downloaded from the web using a python package `bing_image_downloader`
- Total of 10k images (~5000 Korean, ~5000 Indian) were split into Train (80%) and Test (20%) sets. 

(Images here)
Examples of indian cuisine images and korean cuisine images are shown below [Fig x]. As opposed to classifying two specific dishes that are visually very different, classifying cuisine is challenging due to the diversity of dishes in each cuisine.  

Figure below is showing average frequency of R/G/B and G pixel values from all the images in the dataset, and there is no significant difference between the two cuisines. Also, various shapes and textures exists in both cuisines, adding to the complexity of the cuisine recognition.

## Modeling and Evaluation

### Baseline Model
First, a baseline model is established by assigning labels to test data at random while keeping the probabiltiy of each class the same as that of the training set, which is approximately 50%/50% Indian and Korean.

Accuracy is chosen as the metric for model evaulation as we have well balanced classes and the importance of predicting one class properly is not any better or worse than predicting the other class properly. 

With the baseline model, test accuracy is about 50%.

### Multilayer Perceptron
Next, a multilayer perceptron with architecture below gave 95.4% test accuracy. Model was compiled with Adam optimizer with learning rate = 0.0001, Categorical Cross Entropy loss function, were used to optimize the performance. 

### Convoluntional Neural Network 
Lastly, a convolutional nerual network was used to further improve the model, and 96.8% accuracy was achieved. [Figure ]  Dropout layers were added to reduce overfitting and learning rate was adjusted to ----. 


### Examples of incorrectly predicted images:
Some of the incorrectly predicted images were examined:
i) An Indian food (Biryani) was predicted as Korean food: In this case, the two dishes can look very similar, even to a human eye. While recognizing the specific ingredients could be helpful, this information could have been lost from image size reduction (from original to 128 x 128 in preprocessing).
ii) A Korean food (Tteokbokki) was predicted as Indian food: It is a bit unclear what could have contributed to this misclassification.
[Figure]


## Future directions:
In order to improve the prediction accuracy, we could i) incorporate the incorrectly predicted images to training set to improve the model accuracy, ii)add more images(through scraping or augmentation), or iii) increase the model complexity as a next step.





