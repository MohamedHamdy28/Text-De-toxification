# Text-De-toxification

Building a text detoxification model.
- Name: Mohamed Abdelhamid
- Email: m.abdelhamid@innopolis.university
- Group: BS20 AAI

## Basic Usage

### Setting Up the Environment
To set up the necessary environment, follow these steps:

1. Clone the repository.
2. Install the required packages using the command:

   ``` pip install -r requirements.txt ```

## Preparing the Data

To transform the data for use:

1. Navigate to the src/data directory.
2. Execute the make_dataset.py and preprocess_data.py scripts to download and prepare the dataset:

``` python make_dataset.py ```

``` python preprocess_data.py ```

## Training the Model

To train the model:

1. Navigate to the src/models directory.
2. Execute the train_model.py script to start the training process:

```python train_model.py```

This script will train the model and save it to the models/final_solution directory.

## Making Predictions

To make predictions with the trained model:

1. Ensure that the model is trained and saved in the models/final_solution directory.
2. Use the predict_model.py script to detoxify your text:

``` python predict_model.py ```

Modify the prompt variable in the script with the text you wish to detoxify.

