import os
import io
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from google.cloud import vision
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from PIL import Image
import streamlit as st
from google.oauth2 import service_account
import numpy as np
import keyboard

def take_img():
    st.write('Take_img')
    
    # Display the logo and title
    st.image('https://gdm-catalog-fmapi-prod.imgix.net/ProductLogo/eb1b68f8-21ce-4a7e-a7ec-d1adddeba179.png?auto=format&q=50&w=80&h=80&fit=max&dpr=3')
    st.title('Google cloud vision OCR')

    global frame

    cap = cv2.VideoCapture(0)
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    st.title('Webcam Crop')

    x, y, w, h = 100, 100, 450, 250

    w = st.slider('Width', 0, 640, 450)
    h = st.slider('Height', 0, 480, 250)

    stframe = st.empty()
    stop = False
    st.write('Press q to crop')

    while stop == False:
        ret, frame = cap.read()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        stframe.image(frame, channels='BGR')
        if keyboard.is_pressed('q'):
            stop = True
            crop = frame[y:y+h, x:x+w]
            cv2.imwrite('C:/Users/ASUS/Envs/VisionAPIDemo/cap_img/img.jpg', crop)
            break


    cap.release()
    cv2.destroyAllWindows()

    # crop = frame[y:y+h, x:x+w]
    # reset = st.button('Reset')
    # while reset == True:
    #     x, y, w, h = 100, 100, 450, 250
    #     reset = False
    #     break

if st.button ('Take Img'):
    take_img()


# Allow the user to upload an image
# image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

path = 'C:/Users/ASUS/Envs/VisionAPIDemo/cap_img/img.jpg'

if os.path.exists(path):
    image = Image.open(path)

    # option = st.selectbox(
    # 'Select Pre-Processing',
    # ('Original Image','Binary', 'Dilation', 'Erosion', 'Opening', 'Closing'))

    def original_img():
        st.write("Original Image")
        images = [image]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=400)

    def binary_img():
        st.write("Binary")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((3, 3), np.uint8)
        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        Image.fromarray(thres).save("new_img/result.jpg")
        images = [image,thres]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def dilation_img():
        st.write("Dilation")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)

        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        new_img = cv2.dilate(thres, kernel, iterations=1)

        Image.fromarray(new_img).save("new_img/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def erosion_img():
        st.write("Erosion")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)

        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        new_img = cv2.erode(thres, kernel, iterations=1)

        Image.fromarray(new_img).save("new_img/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def opening_img():
        st.write("Opening")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)
        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        new_img = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=1)

        Image.fromarray(new_img).save("new_img/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    def closing_img():
        st.write("Closing")
        im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        kernel = np.array([[1, 0, 0],
                           [0, 1, 0]],np.uint8)

        # #threshold
        ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        new_img = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel, iterations=1)

        Image.fromarray(new_img).save("new_img/result.jpg")
        images = [image,new_img]

        image_on_row = st.columns(len(images))
        for i in range(len(images)):
            image_on_row[i].image(images[i], width=350)

    option = st.selectbox(
    'Select Pre-Processing',
    ('Original Image','Binary', 'Dilation', 'Erosion', 'Opening', 'Closing'))

    if option == 'Original Image':
        original_img()
    elif option == 'Binary':
        binary_img()
    elif option == 'Dilation':
        dilation_img()
    elif option == 'Erosion':
        erosion_img()
    elif option == 'Opening':
        opening_img()
    elif option == 'Closing':
        closing_img()
    

def detect_text(img, credentials_path):

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    if st.button('Detect'):
        with io.open(img, 'rb') as image_file:
            content = image_file.read()
        image = vision_v1.types.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        des = []
        for text in texts:
            des.append(text.description)
        return des

FOLDER_PATH = r'C:\Users\ASUS\Envs\VisionAPIDemo\new_img'
CREDENTIALS_PATH = r'C:\Users\ASUS\Envs\VisionAPIDemo\clever-rite-376610-58e1ce01ea15.json'

st.write("OCR Results:")
st.write(detect_text(os.path.join(FOLDER_PATH, 'result.jpg'), CREDENTIALS_PATH))

# streamlit run webocr.py 