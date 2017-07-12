
# Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/barchart_training.png "Visualization_training"
[image2]: ./examples/barchart_validation.png "Visualization_valid"
[image3]: ./examples/barchart_test.png "Visualization_test"
[image4]: ./examples/normalized.png "normalized"
[image5]: ./examples/download_1_5.png "Traffic Sign 1_5"
[image6]: ./examples/download_6_10.png "Traffic Sign 6_10"
[image7]: ./examples/download_new.png "Traffic Sign new"
[image13]: ./examples/download_new2.png "Traffic Sign new 2"
[image8]: ./examples/demo1.png "possibility_1"
[image9]: ./examples/demo2.png "possibility_2"
[image10]: ./examples/demo3.png "possibility_3"
[image11]: ./examples/demo4.png "possibility_4"
[image12]: ./examples/demo5.png "possibility_5"

---
### Writeup

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python built-in function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data distributes.

![alt text][image1]

The following are two bar chart showing the distribution status of the validation and test data respectively.

![alt text][image2]
![alt text][image3]

From the above tree bar charts presented, I could have the conclution that the tree datasets have the similar sample distribution.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to keep the images as RGB ones because I believed that different color channels might bring more information into this model. The first exploration was that I trained the LeNet-5 model shown in the classroom with the training data. The trainging accuracy was poor and that was showing the model was a little bit underfitting. My strategy was to adjust the model architecture.

As the next step, I normalized the image data because I found that the pixel intensity differs much between samples. The normalization method was to map pixel data in the repective RGB channel to values between -1 to 1. Since such transformation would let the image unable to displayed, it needed to transform the normalized image back to RGB data type just in order to compare the difference before and after the normalization.

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x3 RGB image   				| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| Sigmoid		| 						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 		|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x48   |
| Sigmoid		| 						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 		        |
| Flatten               | outputs 1200                                  |
| Fully connected	| input 1200, output 360        		|
| Sigmoid		| 						|
| Fully connected	| input 360, output 252        		        |
| Sigmoid		| 						|
| Fully connected	| input 252, output 43        		        |
| Softmax		|         					|
|						|												|
|						|												|
 
I tripled the capacity of the original LeNet-5 model, so that the model could learn features provided via different channels.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer provided in Tensorflow. The batch size was set as 128 and the learning rate was 0.001.
As per epochs setting, at first it was set as 10. But the validation accuracy seemed still climbing.
Then it was set as 20 to make sure that the validation accuracy stayed at the peak but not yet started to descend conversely.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.968 
* test set accuracy of 0.950
 
The first architecture that was tried was the original LeNet-5 model shown in the class room. The reason was that its input, output and application was similar to this project, and it was quite easy on the hands.

However, the validation accuracy was poor, and I was wondering whether the model's capacity was not enough. Perhaps its original input was grayscale images, but I choosed to keep the RGB channels.

Therefore the capacity of each layer was tripled. Meanwhile, since all the image data were normalized to the values between -1 to 1, the activating function was replaced with sigmoid function. After such adjusting on the original LeNet-5 model, both training accuracy and validation accuracy were improved significantly.

But the next question was that whether the model was overfitting. I explored different setting for epochs, from 10 to 30, and found that after 15 epochs the trainging accuracy has reached 1.00, but the validation accuracy was still climbing slowly until 20 epochs. Till 30 epochs, the validation accuracy always stayed around the same value, and did not start to descend. No sign of overfitting was shown and the 20 epochs was choosed for the training.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image5] 
![alt text][image6]

The second image might be difficult to classify because the image was not scaled down in proportion. The traffic sign has become oval in shape. Similar distortion was also found on the third, the fifth and the tenth image repectively. 

The sixth image might also be difficult to classify. It was shotted at a different viewing angle from others.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection  	| 
| Roundabout mandatory     			| Roundabout mandatory					|
| Yield						| No passing for vehicles over 3.5 metric tons		|
| Keep right	      		| Keep right					 			|
| Speed limit (60km/h)		| No passing      							|
| Road narrows on the right	| Speed limit (20km/h)      						|
| Priority road			| Priority road      							|
| Turn right ahead		| Yield      								|
| Road work			| Road work      							|
| Stop				| Priority road      							|

The model was able to correctly guess 5 of the 10 traffic signs, which gives an accuracy of 50%.

Such result was quite different from the test data accuracy. The possible reasons might be as the following.

First, the image distortion caused by the scaling. For example, the yield sign image was of 273x185 pixels, instead of 1:1 aspect ratio, and some distortion might be generated while it must be downscaled to 32x32 pixels.

Second, the comparably low amounts of samples in training data. According to the analysis on sign distribution of training data, sign of "road narrows on the right" and "turn right ahead" had much fewer samples. Ths situation might let the model learn less features on these kind of signs, and therefore the model would be unable to classify correctly while facing slightly distorted images.

Furthermore, in order to verify my inference, I clipped all the images to nearly 1:1 aspect ratio. And then the model classified them again. The clipped images are shown below.

![alt text][image7]
![alt text][image13]

This time, except the sign of "Road narrows on the right", all the originally wrong prediction became correct. The Model had 90.0% prediction accuracy on these ten newly clipped images.

Comapred to the 95.0% accuracy on test data set, this result was considerably consistent.
 
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is almost very sure that this is a Right-of-way at the next intersection sign (probability of 0.99), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were shown as below.

![alt text][image8]

For the second to fifth image, the probability is shown as below.

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]



