import streamlit as st
import cv2

def render_header():
    st.write("""
        <p align="center"> 
            <H1> Python Projects
        </p>
    """, unsafe_allow_html=True)

def main():

    st.sidebar.header('Python Projects')
    st.sidebar.subheader('Choose a page to proceed:')
    page = st.sidebar.selectbox("", ["Image Analysis", "Sentiment Analysis", "Emotion Analysis"])

    if page == "Text Detection":
        st.header("Detect and Extract Text From Images")
        st.markdown("""
        **Now, this is probably why you came here. Let's get you some Predictions**
        Upload your Image
        """)
        st.header("Upload Your Image")
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
            
        num_image = np.array(img)
        if st.checkbox('Detect Outlines'):            
            canny = cv2.Canny(num_image, 100, 250)
            st.image(canny, caption='Outline', use_column_width=True, clamp=True)
            edged = cv2.Canny(num_image, 30, 200)
            contour, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            st.info("Count of Contours  = " + str(len(contour)))
            cont = cv2.drawContours(num_image, contour, -30, (0,255,0), 1)
            st.image(cont, caption='Contours', use_column_width=True, clamp=True)

main()
    
    
    
    
    
    
    
    
    
    
    