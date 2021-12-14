# REAL TIME FACE EMOTION RECOGNITION


# Introduction

Communication is one of the most important currencies in which the world is going on.There are many ways of communication.Emotion is a subset of this genre. Emotions are like unspoken truth we see in someone's face. As a social being with conscience, it is easy for a human being to recognize someones emotion from the facial expressions alone.
We want this task to be performed by machines. Deep learning comes into play at this situation.

Artificial Intelligence's rapid development has made a significant contribution to the technological world. Machine Learning (ML) and Deep Learning (DL) algorithms have achieved considerable success in various applications such as classification systems, recommendation systems, pattern recognition etc. 


Emotion recognition is very significant as it has wide range applications in the field of Computer Vision and Artificial Intelligence.


 
![download](https://user-images.githubusercontent.com/88419896/146048598-362d373a-f5a8-42d1-8ade-5f5b282ccd3c.png)
 
 
 # Problem Statement
 
 The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms.

Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge.

In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention. Digital classrooms are conducted via video telephony software program (exZoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to lack of surveillance. While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analysed using deep learning algorithms. Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analysed and tracked.


We will solve the above-mentioned challenge by applying deep learning algorithms to live video data. The solution to this problem is by recognizing facial emotions.

This is a few shot learning live face emotion detection system. The model should be able to real-time identify the emotions of students in a live class.

![image](https://user-images.githubusercontent.com/88419896/146051028-83114199-a076-4954-9744-1a4fdd941ab8.png)


# Model Creation

## 1) Custom Model 1

Model 1 is sequential model which uses the ELU activation function.

![image](https://user-images.githubusercontent.com/88419896/146051656-228ea429-ea80-4d0c-ac06-2536a13bd902.png)

Training gave 78 percentage accuracy and validation gave 71 percentage accuracy


## 2) Custom Model 2

Model 2 is also sequential model which uses RELU activation function.

![image](https://user-images.githubusercontent.com/88419896/146051721-39e23d3e-3822-488d-8f1a-30404c7988b3.png)

Training gave 72 percentage accuracy and validation gave 68 percentage accuracy.

As the model1 has the better accuracy among the two, this model is deployed in the streamlit application for emotion recognition


# Real Time Emotion Recognition

In this repository I have made a front end emotion recognition application using streamlit .This model was deployed on heroku also.

Streamlit Link :- https://share.streamlit.io/nayanapradeep/face-emotion-recognition/main/app.py

Heroku Link :- https://emotion-recognition-2021.herokuapp.com/


Steps for hosting this application in local environment
1) Create a new virtual environment with python 3.7 version
2) Install streamlit,tensorflow,opencv,streamlit-webrtc packages 
3) Download the app.py, model1.h5, model1.json and haarcascade_frontalface_default.xml from this repository to a local folder in your machine
4) In the new virtual environment using command prompt go to the app.py folder location
5) run the command : streamlit run app.py

The application will open up in your local browser.

Note: Sometimes you have to refresh 2-3 times to get the app running successfully


# Conclusion

Emotion Recognition Application has been deployed and providing accurate predictions.

Demo of this application is provided in the link: https://github.com/NayanaPradeep/Face-Emotion-Recognition/tree/main/Demo






