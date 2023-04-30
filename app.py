import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
import webbrowser

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

def openCamera():
    # start the webcam feed
    cap = cv2.VideoCapture(0)

    # pass here your video path
    # you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
    #cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            print(emotion_dict[maxindex])
        
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            emotion_return = emotion_dict[maxindex]
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion_return

#UI
st.set_page_config(page_title="Music Recommender")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
#st.image("header.jpg", caption="")
st.title("Emotion based Music Recommender")
language=st.text_input("Enter the language")
singer=st.text_input("Singer Name")
col1,col2 = st.columns(2)
with col1:
    btn=st.button("Open YouTube Music")
with col2: 
    btn2 = st.button("Open SoundCloud")
st.write("Our project is an emotion-based music recommendation system. The user inputs their preferred language and the name of a singer, along with their facial expression through a camera. The system then detects the user's emotion and recommends a suitable music track based on that emotion. The output is the suggested music track that the user can listen to. The project combines elements of natural language processing, computer vision, and music recommendation algorithms to create a personalized and interactive experience for the user.")
if(btn):
    st.snow()
    emotion=openCamera()
    webbrowser.open(f"https://music.youtube.com/search?q={singer}+{language}+{emotion}+songs")
if(btn2):
    st.snow()
    emotion=openCamera()
    webbrowser.open(f"https://soundcloud.com/search?q={singer}+{language}+{emotion}+songs")


