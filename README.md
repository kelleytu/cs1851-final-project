# cs1851-final-project

MODEL ARCHITECTURE:
- Uses an intermediate-fusion approach to combine image and tabular data.

UPDATES FROM MIDTERM CODE SUBMISSION:
In the midterm code submission, we implemented a late-fusion model which took a weighted average of the CNN image predictions with the Gradient Boosting Classifier tabular data predictions. Since then, we have made the following improvements (in chronological order):

- Implemented further preprocessing steps based on other papers:
   - Hair removal (with and without erosion) identifies edges and masks them. Erosion degrades the edges such that only the stronger/bolder edges are removed; however, we found that no erosion was able to better capture more of the hairs.
   - Color constancy: adjusts color and lighting across samples to make each image have more standardized color balance
   - Center cropping: we noticed gray borders around some images and circular black borders around others, so we center cropped the images to remove these artifacts, resizing the image to 204x204 (originally 224x224)
- Intermediate fusion with CNN + MLP classifier: concatenating CNN flattened output with tabular data, then classifying with MLP. Performed similarly to base CNN.
- Added ResNet image classifier:
   - Started by loading and implementing an image classifier with just the ResNet architecture, without the pre-trained weights. We found that test_f1 was slightly higher than our regular CNN model, but not meaningfully better.
   - Then, loaded pre-trained weights alongside ResNet architecture and found that test_f1 was significantly better even after just one epoch. Additionally, the ResNet model trained faster.
- Added class weights to address heavy class imbalance in the training data. This was especially relevant because our evaluation emphasized macro performance, where poor performance on rare classes would significantly reduce test_f1 score. After computing class weights in training and adding it to the loss function, validation f1 became much more balanced across classes, signaling that the model was not predicting dominant labels as much.
- Switched optimizer from Adam to AdamW to apply weight decay without hindering optimal convergence and assign separate learning rates:
   - We used AdamW to decouple weight decay from gradient updates, so that it is applied directly to parameter updates and not the loss function, leading to improved generalization. We found that the model performed significantly better on out-of-sample data when it came to test_f1 score.
   - We assigned a smaller learning rate for the pre-trained ResNet model and a larger learning rate for the tabular classifier. The smaller learning rate for ResNet allowed helpful features from the pre-trained model to remain relevant.
- Added simple data augmentation consisting of random vertical and horizontal flips. However, this likely had little effect since lesions are already mostly elliptical and have no directionality.
- Image cropping: instead of just cropping the 224x224 image to 204x204 (as stated above), we instead cropped to 204x204 and then added padding to restore the standard 224x224 ResNet image size. 
   - Initially determined padding color by median pixel value of the entire image, but found that images with very large lesions caused the border to be miscolored.
   - Thus, we instead determined padding color using the median pixel value of the outer 20 pixels of the image, which, upon analysis, did a much better job of identifying the "normal" background/skin color.
- Implemented scheduler to reduce learning rates when performance plateaus. We did not want f1 change to be too aggresive, so factor was set to 0.5 and patience to 3. Although patience of 3 seems fairly small, we were running a max of 20 epochs and often observed plateauing around epochs 10-15.  
- Modified loss function to include focal loss in addition to cross entropy loss. Default gamma was used to encourage model to focus on harder (lower confidence) samples.
- Implemented Test Time Augmentation to try to improve submission F1 performance. We found that the model struggled to generalize when moving from training/validation to testing on out-of-sample data. To combat this, we tried averaging the predictions of original preprocessed images, horizontally-flipped images, vertically-flipped images, and horizontal+vertical-flipped images. We compared the value counts of each label before and after Test Time Augmentation and found that there was a slight difference in how the model classified each prediction.

CODE FOR ANALYSIS AND INTERPRETATION:
- Added plotting of model loss, accuracy, and f1 across epochs. Was helpful for examining overfitting and determining number of epochs for future runs
- Added generation confusion matrix of best model (based on f1). Best model is saved per run based on epoch f1. Confusion matrix revealed extreme class imbalance (significantly more class 4) and common misclassifications. For example, class 4 was ofen misclassified as class 0 and 3.
- Added plotting of embeddings. Embeddings include image features, tabular features, and concatenated image and tabular features. Allows us to visualize learned representations. Embeddings revealed overlap of representations of class 0, 3, and 4, verifying confusion matrix observations.
- Added Grad-CAM (https://github.com/jacobgil/pytorch-grad-cam), which produces visual heatmaps of the regions of the image that are most important to the CNN output.


USAGE:
- python run_model.py
   - Trains model and saves model data and metrics. Model data and metrics are saved in new folder labelled based on number of epochs used. Epoch number is hard coded in run_model.py. Transform preprocessor may be saved in the current working directory and should be moved into the saved_model/{model_folder}
- python analyze_results.py saved_model/20
   - Plots model loss, accuracy, and f1 across epochs. Plots embeddings and displays Grad-CAM analysis of sample image. Model used is best model based on test dataset.
