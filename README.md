# American Sign Language Recognition with Deep Learning

Sign languages are a group a communication languages that use a combination of manual articulations in combination with non-manual elements to convey messages. There are different sign language with variability in hand shape, motion profile, and position of the hand, face, and body parts contributing to each sign. Each country generally has its own native sign language, and some have more than one: the current edition of [Ethnologue](https://www.ethnologue.com/subgroups/sign-language) lists 144 sign languages.

The simplest class of sign languages, know as fingerspelling systems is limited to a collection of manual sign that representing the symbol of an alphabet.

Sign language recognition and translation is one of the application areas of human computer interaction (HCI) where signs are recognized and then translated and converted to text or voice and at symbol, word or sentence level

### The Sign Language Recognition Problem
The general problem include three different tasks:

1. static o continuous letter/number gesture recognition (classification problem)
2. static or continuous single word recognition (classification problem)
3. sentence level sign language recognition (Natural Language Processing problem)

In this demo project we will create two different models to accomplish  the first task.

## Table Of Contents

- [American Sign Language Recognition with Deep Learning](#american-sign-language-recognition-with-deep-learning)
    - [The Sign Language Recognition Problem](#the-sign-language-recognition-problem)
  - [Table Of Contents](#table-of-contents)
  - [Project Workflow](#project-workflow)
  - [Project Set Up and Installation](#project-set-up-and-installation)
  - [Dataset](#dataset)
    - [Overview](#overview)
    - [Task](#task)
    - [Access](#access)
  - [Automated ML](#automated-ml)
    - [Results](#results)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Results](#results-1)
  - [Model Deployment](#model-deployment)
  - [Screen Recording](#screen-recording)
  - [Standout Suggestions](#standout-suggestions)

## Project Workflow

In this project, we will create two models: one using Azure Automated ML (_AutoML_ from now on) and one customized model whose hyperparameters are tuned using HyperDrive. Then we will compare the performance of both the models and deploy the best performing model as web service using AzureML.

![img](media/project_workflow.png)

All steps are described int this document and a screencast that shows the processs and the final working ML application is provided.


## Project Set Up and Installation
To run this project you will neeed an Azure ML workspace.
The provided Jupyter notebboks can be executed in local development environment or inside Azure Machine Learning Studio.

In the local development enviroment with the following libraries installed:

- Numpy and Pandas — for data handling and processing
- PyTorch and Torchvision — for machine learning functions
- Matplotlib — for data visualization
- Azure ML SDK
- Azure ML widget extention installed (if notebook is executed from Visual Studio Code)


```
!conda install numpy pytorch torchvision cpuonly -c pytorch -y
!pip install matplotlib --upgrade --quiet
!pip install azureml-sdk
```

The custom model can also be trained running the notebbok inside AzureML workspace. 

## Dataset
The [American Sign Language MNIST Dataset from Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist) is used for this project. A copy of the dataset 

![MNIST ASL Dataset](datasets/sign-language-mnist/amer_sign3.png)

### Overview

This dataset is in tabular format and is similar to the original MNIST dataset.

Each row in the csv file represents a label and a single 28x28 pixel greyscale image represented using 784 pixel values ranging from 0-255

The label in the dataset is a number ranging from 0-25 associated with its english letter equivalent (e.g. 0 = a)
 
There is no label correspondence to the letter J (9) and Z (25) due to the motion required to symbolize those letters. 

In total there are 27,455 training cases and 7,172 tests cases in this dataset.


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.



### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

- Use [Dataset Monitor](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets?tabs=python) to detect data drift
- Publish a pipeline for automatic model re-training
- 

