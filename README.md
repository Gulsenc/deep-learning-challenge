# deep-learning-challenge

Overview of the Analysis
The goal of this analysis was to create and test a deep learning model that can predict whether charity applications will be successful or not. The main objective was to build a model that could accurately predict outcomes with an accuracy rate higher than 75%. This model would help Alphabet Soup make better decisions based on the information available in the dataset.

Results
Data Preprocessing

Target Variable(s):
The target variable in this model is IS_SUCCESSFUL. This variable shows whether a charity application was successful (1) or not (0).

Feature Variable(s):
The features used in the model include all the columns in the dataset, except for EIN, NAME, and IS_SUCCESSFUL. These features represent different aspects of the charity applications, like APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, and ORGANIZATION. These categorical variables were converted into numerical form using one-hot encoding.

Removed Variable(s):
The EIN and NAME columns were removed because they don’t help in predicting the success of applications. EIN is just an identifier for each application, and NAME is a text field that doesn’t help the model in making predictions.

Compiling, Training, and Evaluating the Model
Neurons, Layers, and Activation Functions:

The neural network model started with three hidden layers: (the starter.ipynb)
First Hidden Layer: 100 neurons with the ReLU activation function.
Second Hidden Layer: 50 neurons with the ReLU activation function.
Third Hidden Layer: 25 neurons with the ReLU activation function.
The ReLU activation function was chosen because it adds non-linearity to the model and avoids problems like the vanishing gradient, which can happen with other functions like sigmoid or tanh.

Model Performance:

The first model reached an accuracy of about 72.79%, which is below the target of 75%.
To improve this, several optimization steps were taken: (AlphabetSoupCharity_Optimized.ipynb)
Increased Neurons and Added Layers: The model was modified to have more neurons and extra layers to improve its ability to learn.
Implemented Dropout and Batch Normalization: Dropout layers were added to prevent overfitting, and batch normalization layers were added to help the model train more smoothly.
Adjusted Learning Rate: The learning rate was changed using a scheduler to help the model learn better.
Even with these changes, the optimized model reached an accuracy of about 72.83%, which is only a small improvement and still below the target.

Summary
The deep learning model created for Alphabet Soup was able to predict the success of charity applications with an accuracy of around 72.83%. Although this result is good, it did not meet the goal of 75% accuracy. Several techniques were used to try and improve the model, such as making the model more complex, adding dropout layers, and adjusting the learning rates.

Recommendation:
To further improve the model, it could be helpful to try other machine learning algorithms, like Random Forest, Gradient Boosting, or Support Vector Machines. These methods sometimes work better than neural networks for structured data tasks, especially when the dataset is not very large or when careful feature engineering is important. 
