# My_Protein_Tracker_AI

This project presents an AI tool designed to detect the protein content in various food items, including fruits, vegetables, nuts, and meats. The tool leverages a comprehensive dataset covering various foods to provide accurate protein content information.

Whether you're a gym rat meticulously tracking your macros, a greens enthusiast who still wants protein, a new mother following a keto diet, or simply curious about the nutritional value of your meals, this AI tool offers a convenient solution. It eliminates the hassle of searching through multiple sources and tabs on Google, allowing you to quickly and easily find the protein content of your meals in one place.

Here is a preview:

![A preview of what the protein detection looks like](https://github.com/incheresting/nvidiaproject/assets/139397694/d2caa99e-b37f-4675-846b-6233f36c934a)

## The Algorithm

Components Used:

1. imagenet.py: Supports viewing the live video feed from the webcam.
2. videobase.py: Provides the foundational functionality for handling video inputs.
3. my-recognition.py: Developed to utilize image detection to classify the food item in real-time

How It Works:
Live Video Feed: When the webcam is focused on a specific food item (this set-up process is through videobase), imagenet.py manages the live video feed, ensuring smooth and continuous visual input.
Image Classification: Resnet-18 (through my recognition) is the core model for classifying food items.


## Running this project

1. Ensure these are imported:
![Imports that are needed](https://github.com/incheresting/nvidiaproject/assets/139397694/fc374a59-dbe4-4ee5-a58f-d144a3726d65)
** The program can not work without resnet-18 and imagenet
2. To start a live video feed, type "python3 my-recognition.py /dev/video0 [desired file name].mp4"
3. Ensure the webcam is set up to capture food items accurately
4. When the image is correctly classified, type ctrl + C
![What the classification process looks like](https://github.com/incheresting/nvidiaproject/assets/139397694/ad197f88-bca3-4bb4-8802-77652c259faf)
5. Download the file of [desired file name] and open. 
https://github.com/incheresting/nvidiaproject/assets/139397694/ce3f19a9-d269-4070-a873-1e58427e06da

[View a video explanation here](video link)
