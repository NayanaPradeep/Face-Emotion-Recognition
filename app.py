# Importing Libraries
import numpy as np
import cv2
import av
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# load model
emotion_dict = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

classifier =load_model('model1.h5')

# load weights into new model
classifier.load_weights("model1.h5")


#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoProcessorBase):
    def recv(self,frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Face Analysis Application #
    st.title("FACE EMOTION RECOGNITION")
    sel_choice = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", sel_choice)
    st.sidebar.markdown(
        """ DEVELOPED BY 
	      NAYANA PRADEEP
 	      nayanapradeep92@gmail.com""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#9D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion Recognition Application using Custom CNN model </h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time face detection using web cam feed.

                 2. Real time face emotion recognization.

                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("CLICK ON START TO BEGIN THE DETECTION")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    elif choice == "About":
         
        html_temp_about= """<div style="background-color:#9D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real Time Face Emotion Recognition using Custom CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about, unsafe_allow_html=True)
 
    
    else:
        pass


if __name__ == "__main__":
    main()