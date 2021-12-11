from typing import List, Any
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from fastai import *
from fastai.vision import *
import io
import streamlit.components.v1 as components
from streamlit_webrtc import ClientSettings, VideoTransformerBase, WebRtcMode, webrtc_streamer
import av
from typing import Union
import classify
# import logging
import logging.handlers
import threading
from pathlib import Path

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.unsplash.com/photo-1551523713-c1473aa01d9f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=701&q=80")
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""<style>body{
# backgroundColor="#9fb2d8";
# secondaryBackgroundColor="#6072e0";
# textColor="#e807bc";}</body></style>""", unsafe_allow_html=True)

st.title('Welcome to Face Emotion Recognition Application')

main_options = st.selectbox("What would like to choose: ", ['About', 'Detection space', 'Contact'])

if main_options == 'About':
    st.write('This is the solution to real time face emotion detection problem')
    st.write('Today, the majority of our time is spent on interacting with computers and mobile phones in our daily life due to \
    technology progression and ubiquitously spreading these mediums. However, they play an essential role in our \
    lives, and the vast number of existing software interfaces are non-verbal, primitive and terse. Adding emotional\
     expression recognition to expect the users’ feelings and emotional state can drastically improve human–computer interaction ')
    st.write('Humans usually employ different cues to express their emotions, such as facial expressions, hand \
    gestures and voice. Facial expressions represent up to 55% of human communications while other ways such as \
    oral language are allocated a mere 7% of emotion expression. Therefore, considering facial expressions in an \
    HRI(Human robotic interactions system enables simulation of natural interactions successfully')
    st.write('This has many more advantages than we can imagine')
    st.write('This can be used in education, research, medicine, manufacturing, investigation and many more')
    st.markdown(
        '### How good it is that the people are able to catch the things or understand properly, Same with real time detection')
    st.write('This when used in right way gives many good results')
    st.write('Note: This was created with the objective of using in field of education which detect the emotions\
     of the students which enables in proper understanding and teaching the students the right way to do the things.\
     Works even like surveillance camera which keeps the eye on students emotions')

elif main_options == 'Contact':
    st.write('''
                      nayan92@gmail.com
                      ''')
elif main_options == 'Detection space':

    option = st.radio('Which type of detection would you like to make?',
                      ('an Image', 'a Video', 'OpenCV Live', 'an Instant Snapshot live detection',
                       'a Live Video detection'))
    st.header('You selected {} option for emotion detection'.format(option))

    if option == 'an Image':

        uploaded_file = st.file_uploader("Choose an image", type=['jpg'])
        if uploaded_file is not None:
            # image2 = Image.open(uploaded_file)
            # st.write('Image 2')
            # st.write(type(image2))
            # st.image(image2, caption='Uploaded Image', use_column_width=True)
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Now do something with the image! For example, let's display it:
            st.image(opencv_image, channels="BGR", use_column_width=True)
            # st.write(type(opencv_image))
            gray1 = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            # st.write(type(gray1))

            if st.button('Detect the Emotion'):
                st.write("Result...")
                st.write("Model Loaded")
                model4 = classify.get_model()
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

                # read the processed image then make prediction and display the result

                t = pil2tensor(opencv_image, dtype=np.float32)  # converts to numpy tensor
                t = t.float() / 255.0
                img1 = Image(t)

                # Convert to fastAi Image - this class has "apply_tfms"

                model_pred1 = model4.predict(img1)[0]

                st.write(str(model_pred1))
                model_pred2 = classify.prediction(img1)

                st.subheader('Detection made by fastai model using streamlit: {}'.format(str(model_pred1).capitalize()))

    elif option == 'a Video':
        st.subheader("Play the Uploaded File while detecting")

        uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
        temporary_location = False

        if uploaded_file is not None:
            g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
            temporary_location = "testout_simple.mp4"

            with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file

            # close file
            out.close()


        @st.cache(allow_output_mutation=True)
        # @st.cache(suppress_st_warning=True)
        def get_cap(location):
            print("Loading in function", str(location))
            video_stream = cv2.VideoCapture(str(location))

            # Check if camera opened successfully
            if video_stream.isOpened() == False:
                print("Error opening video file")
            return video_stream


        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        scaling_factorx = 0.25
        scaling_factory = 0.25
        image_placeholder = st.empty()


        # test to run until we stop or video ends

        def test_rerun(text, video_stream):
            while True:
                ret, image = video_stream.read()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, "The last phase of the person's Emotion was recorded " + str(text), (95, 30), font,
                            1.0, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(image, "Press SPACE: Detecting", (5, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(image, "Hold Q: To Quit😎", (460, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for x, y, w, h in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.imshow("Image", image)

                if cv2.waitKey(1) == ord(' '):
                    cv2.imwrite("test7.jpg", image)
                    model5 = classify.get_model()
                    # st.write('Model Loaded')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # read the processed image then make prediction and display the result
                    t = pil2tensor(image, dtype=np.float32)  # converts to numpy tensor
                    t = t.float() / 255.0
                    img1 = Image(t)
                    text = model5.predict(img1)[0]
                    text = str(text)
                    # text
                    print(text)
                    # st.write(text)
                    test_video_pred(text)
                    break

                if cv2.waitKey(1) == ord('q'):
                    video_stream.release()
                    cv2.destroyAllWindows()
                    break
            return text


        @st.cache(allow_output_mutation=True)
        # @st.cache(suppress_st_warning=True)
        def test_video_pred(text):
            if temporary_location:
                while True:
                    # here it is a CV2 object
                    video_stream = get_cap(temporary_location)
                    # video_stream = video_stream.read()
                    ret, image = video_stream.read()
                    if ret:
                        image = cv2.resize(image, None, fx=scaling_factorx, fy=scaling_factory,
                                           interpolation=cv2.INTER_AREA)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(image, "The last phase of person's emotion was recorded: " + str(text), (95, 30),
                                    font,
                                    1.0,
                                    (255, 0, 0), 2, cv2.LINE_AA)

                        cv2.putText(image, "Press SPACE: For detection", (5, 470), font, 0.7, (255, 0, 0), 2,
                                    cv2.LINE_AA)

                        cv2.putText(image, "Hold Q: To Quit😎", (460, 470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        for x, y, w, h in faces:
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        cv2.imshow("Image", image)

                        if cv2.waitKey(1) == ord(' '):
                            cv2.imwrite("test7.jpg", image)
                            model5 = classify.get_model()
                            # st.write('Model Loaded from test_video_pred')
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                            # read the processed image then make prediction and display the result

                            t = pil2tensor(image, dtype=np.float32)  # converts to numpy tensor
                            t = t.float() / 255.0
                            img1 = Image(t)
                            text = model5.predict(img1)[0]
                            text = str(text)
                            # st.write(text)
                            # text
                            print(text)
                            # st.write(str(text))
                            test_rerun(text, video_stream)
                            # plt.imshow(img)
                            break

                        if cv2.waitKey(1) == ord('q'):
                            video_stream.release()
                            cv2.destroyAllWindows()
                            break
                    else:
                        print("there was a problem or video was finished")
                        cv2.destroyAllWindows()
                        video_stream.release()
                        break
                    # check if frame is None
                    if image is None:
                        print("there was a problem None")
                        # if True break the infinite loop
                        break

                    image_placeholder.image(image, channels="BGR", use_column_width=True)

                    cv2.destroyAllWindows()
                video_stream.release()

                cv2.destroyAllWindows()
            return text


        test_video_pred('None')

    elif option == 'OpenCV Live':
        st.write('This is not a good option for streamlit then it works well locally')
        st.write('OpenCV is popular python library used when images and videos are involved')
        st.write('OpenCV is a good option for computer vision problems')
        st.markdown(
            '### For real time detection in streamlit please use a Live Video detection option/an Instant Snapshot detection for finding out the emotion of a person in live')
        st.write(
            "Streamlit doesn't support OpenCV for live detection for some reasons and webrtc solves this problem in streamlit. Hence choose the other options for detection")
        st.write('Thanks for reading')
        st.write('Thank you')

    elif option == 'an Instant Snapshot live detection':
        def transform(self, frame):
            label = []
            img = frame.to_ndarray(format="bgr24")
            face_cascade_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade_detect.detectMultiScale(gray, 1.3, 1)

            for (x, y, w, h) in faces:
                a = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
                t = pil2tensor(roi_gray, dtype=np.float32)  # converts to numpy tensor
                t = t.float() / 255.0
                roi = Image(t)
                model6 = classify.get_model()
                prediction = model6.predict(roi)[0]  # Prediction
                label = str(prediction)
                label_position = (x, y)
                b = cv2.putText(a, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return b


        class VideoTransformer(VideoTransformerBase):
            pass


        def face_detect():
            class VideoTransformer(VideoTransformerBase):
                frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
                in_image: Union[np.ndarray, None]
                out_image: Union[np.ndarray, None]

                def __init__(self) -> None:
                    self.frame_lock = threading.Lock()
                    self.in_image = None
                    self.out_image = None

                def transform(self, frame: av.VideoFrame) -> np.ndarray:
                    in_image = frame.to_ndarray(format="bgr24")
                    out_image = in_image[:, ::-1, :]  # Simple flipping for example.

                    with self.frame_lock:
                        self.in_image = in_image
                        self.out_image = out_image

                    return in_image

            ctx = webrtc_streamer(key="snapshot", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer,client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)

            while ctx.video_transformer:

                with ctx.video_transformer.frame_lock:
                    in_image = ctx.video_transformer.in_image
                    #out_image = ctx.video_transformer.out_image

                if in_image is not None:
                    gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
                    face_cascade_detect = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade_detect.detectMultiScale(gray)
                    for (x, y, w, h) in faces:
                        a = cv2.rectangle(in_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_gray = cv2.resize(roi_gray, (48, 48),
                                              interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
                        if np.sum([roi_gray]) != 0:
                            t = pil2tensor(roi_gray, dtype=np.float32)  # converts to numpy tensor
                            t = t.float() / 255.0
                            roi = Image(t)

                            # roi = np.expand_dims(roi, axis=0)  ## reshaping the cropped face image for prediction
                            model6 = classify.get_model()
                            prediction = model6.predict(roi)[0]  # Prediction
                            label = str(prediction)
                            label_position = (x, y)
                            b = cv2.putText(a, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                            2)  # Text Adding
                            st.image(b, channels="BGR")
                        else:
                            st.write('Unable to access camera input')


        HERE = Path(__file__).parent

        logger = logging.getLogger(__name__)

        WEBRTC_CLIENT_SETTINGS = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": True},
        )
        face_detect()
        st.write('To load the live video it might take some time please wait for few minutes and the detection begins')
        st.write('This is the end of the instructions for using this option. See you')
        st.write(
            'Snap is taken and emotion is detected immediately whereas even without snap it is possible and there will be some lag. You can directly go to live video detection option')
        # st.write('Live snapshot detection')
    elif option == 'a Live Video detection':
        class VideoTransformer(VideoTransformerBase):
            def transform(self, frame):
                global b
                label = []
                img = frame.to_ndarray(format="bgr24")
                face_cascade_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                # emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade_detect.detectMultiScale(gray, 1.3, 1)
                for (x, y, w, h) in faces:
                    a = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
                    t = pil2tensor(roi_gray, dtype=np.float32)  # converts to numpy tensor
                    t = t.float() / 255.0
                    roi = Image(t)
                    model6 = classify.get_model()
                    prediction = model6.predict(roi)[0]  # Prediction
                    label = str(prediction)
                    label_position = (x, y)
                    b = cv2.putText(a, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return b


        def live_detect():
            class VideoTransformer(VideoTransformerBase):
                frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
                in_image: Union[np.ndarray, None]
                out_image: Union[np.ndarray, None]

                def __init__(self) -> None:
                    self.frame_lock = threading.Lock()
                    self.in_image = None
                    self.out_image = None

                def transform(self, frame: av.VideoFrame) -> np.ndarray:
                    in_image = frame.to_ndarray(format="bgr24")
                    out_image = in_image[:, ::-1, :]  # Simple flipping for example.

                    with self.frame_lock:
                        self.in_image = in_image
                        self.out_image = out_image

                    return in_image

            ctx = webrtc_streamer(key="snapshot", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)
            while ctx.video_transformer:

                with ctx.video_transformer.frame_lock:
                    in_image = ctx.video_transformer.in_image
                    # out_image = ctx.video_transformer.out_image
                if in_image is not None:
                    gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
                    face_cascade_detect = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade_detect.detectMultiScale(gray)
                    for (x, y, w, h) in faces:
                        a = cv2.rectangle(in_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_gray = cv2.resize(roi_gray, (48, 48),
                                              interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
                        if np.sum([roi_gray]) != 0:
                            t = pil2tensor(roi_gray, dtype=np.float32)  # converts to numpy tensor
                            t = t.float() / 255.0
                            roi = Image(t)
                            # roi = np.expand_dims(roi, axis=0)  ## reshaping the cropped face image for prediction
                            model6 = classify.get_model()
                            prediction = model6.predict(roi)[0]  # Prediction
                            label = str(prediction)
                            label_position = (x, y)
                            b = cv2.putText(a, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Text Adding
                            st.image(b, channels="BGR")
                else:
                    st.write('Unable to access camera input')


        HERE = Path(__file__).parent

        logger = logging.getLogger(__name__)

        WEBRTC_CLIENT_SETTINGS = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": True})

        # class VideoTransformer(object):
        #   pass
        #live_detect()

        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, mode=WebRtcMode.SENDRECV, client_settings=WEBRTC_CLIENT_SETTINGS, async_transform=True)

        # live_detect()
        st.write('Live functioning')
        st.write(
            'This is running using newly introduced webrtc tool which can access the camera whereas opencv cannot function properly in streamlit')
        st.write(
            'This new tool takes some time for starting up the live video,please wait for few minutes(2-3 mins max) and the detection starts')
        st.write('This is the end of the instructions for using this option. See you')

    else:
        st.write('You did not select the proper option as specified. Please select a valid option')
        st.write(
            'If one of the four options was selected and it did not work. Please clear the cache and rerun the application')
        st.write('Thanks for understanding')

st.write('Thank you. I hope you got emotions detected which are hidden in the picture or an image or a video')
st.write('See you soon')
 