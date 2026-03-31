# cs1851-final-project

MODEL ARCHITECTURE:
- Uses late fusion approach to combine outputted probabilities of CNN image classification and gradient boosting classification models
    - CNN: performs classification based on train_images; uses dropout and max pooling
    - Gradient Boosting Classifier: performs classification based on train_tabular; uses tuned hyperparameters to optimize for the best F1 performance
- To combine the two models, we calculated the probabilities of each class on the test dataset, then used a weighted average of the probabiities outputted by each model to obtain the final predictions

RESULTS:
- Observed that the CNN had better accuracy than the Gradient Boosting Classifier but performed worse on all other metrics
- For the ensemble model, we observed that all metrics were the same or better than both models individually, suggesting that multimodal learning may improve performance across metrics

NEXT STEPS:
- Work on preprocessing the images, implementing color constancy, hair removal, and data augmentation
- Incorporate further preprocessing on the tabular data (handling for missingness, duplicates, etc.) 
- Further tune hyperparameters for Gradient Boosting model
- Test other model architectures to include in our ensemble model
- Consider early or intermediate fusion of multiple modalities and/or perform further computation to determine the optimal alpha value (hyperparameter which determines relative weighting of CNN vs gradient boosting model)