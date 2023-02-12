
![GitHub repo size](https://img.shields.io/github/repo-size/tylerchi1000/StreamlitObjectDetection)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tylerchi1000-streamlitobjectdetection-home-z5ovvy.streamlit.app/)
Badge [source](https://shields.io/)

# Raspberry Object Detection

Train and detect bounding boxes for raspberries and app for streamlit webservice
[Link to Streamlit](https://tylerchi1000-streamlitobjectdetection-home-9ldy0p.streamlit.app/)

![Screenshot](https://github.com/tylerchi1000/StreamlitObjectDetection/blob/main/assets/Home_screenshot.jpg)

## Authors

-[tylerchi1000](https://github.com/tylerchi1000)

## Table of Contents

 - [Business application](#business-application)
 - [Data source](#data-source)
 - [Methods](#methods)
 - [Tech Stack](#tech-stack)
 - [Training](#training)
 - [App deployment on Streamlit](#app-deployed-on-streamlit)
 - [Limitations and potential improvements](#Limitations-and-potential-improvements)

## Business application

This app uses object detection to create bounding boxes for both ripe and unripe raspberries. It can be used for labeling assistance and crop counting applications.

## Data source

I self labeled images and hosted them on [RoboFlow](https://app.roboflow.com/tyler-chinn-xnddb/fruit-detection-sample/3)

## Tech Stack

-Python
-Streamlit
-RoboFlow
-YOLOv8

## Training

If you would like to train your own model follow these steps...

 - Find a dataset or create your own. Convert to [YOLOv8 label format](https://roboflow.com/formats/yolov8-pytorch-txt)
 - Use the RaspberryTrainFinal.ipynb file to setup a Google Colab Environment
 - Load images in the correct [file format](https://blog.roboflow.com/how-to-save-and-load-weights-in-google-colab/)
 - Run through training notebook and retrieve best.pt to replace one in respository
 - Change Home.py box_label function to include your number of classes if desired (dataframe output should not require updates)

## App deployment on Streamlit

 - Create a Streamlit account or login with Github, Google, etc.
 - Click on new app button
 - Insert Github URL, branch = main, python file = Home.py
 - Click advanced settings and select python version 3.8
 - Deploy

## Limitations and potential improvements

 - Trained on a very small custom made dataset. While I did some data augmentation more samples would likely increase model performance.
 - The sample size is skewed towards ripe berries
 - Performance on berries in the background (out of focus) is more limited
