import streamlit as st
from PIL import Image
import cv2
from pytesseract import pytesseract

def render_header():
    st.write("""
        <p align="center"> 
            <H1> Python Projects
        </p>
    """, unsafe_allow_html=True)

def main():

    st.sidebar.header('Python Projects')
    st.sidebar.subheader('Choose a page to proceed:')
    page = st.sidebar.selectbox("", ["Text Detection", "Sentiment Analysis", "Emotion Analysis"])

    if page == "Text Detection":
        st.header("Detect and Extract Text From Images")
        st.markdown("""
        **Now, this is probably why you came here. Let's get you some Predictions**
        Upload your Image
        """)
        st.header("Upload Your Image")
        path_to_tesseract = r"tesseract.exe"
        pytesseract.tesseract_cmd = path_to_tesseract

        file_path = st.file_uploader('Upload an image', type=['png', 'jpg'])

        if file_path is not None:
            img = cv2.imread(file_path)
            st.success('File Upload Success!')
        else:
            st.info('Please upload Image file')

        if st.checkbox('Show Uploaded Image'):
            st.info("Showing Uploaded Image ---->>>")
            st.image(img, caption='Uploaded Image',
                     use_column_width=True)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 5)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = img.copy()
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw the bounding box on the text area
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.image(rect, caption='Sample Data', use_column_width=True)
            # Crop the bounding box area
            cropped = im2[y:y + h, x:x + w]
            # Using tesseract on the cropped image area to get text
            text = pytesseract.image_to_string(cropped)
            st.write("The texts are: ",text)

main()
    
    
    
    
    
    
    
    
    
    
    