import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image

st.title("Automated Resume Screener 📄")
st.write("Easily check the match percentage between a Resume and a Job Description.")

jd_text = st.text_area("Paste the Job Description (JD) here:", height=150)

resume_text = st.text_area("Paste the Candidate's Resume text here:", height=150)

st.header("OCR - Extract Text from Resume Image")
uploaded_file = st.file_uploader("Upload your Resume Image (PNG, JPG)", type=["png", "jpg", "jpeg"])

extracted_text = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Resume Image', use_container_width=True)
    
    with st.spinner("Extracting text from the image..."):
        extracted_text = pytesseract.image_to_string(image)
        st.subheader("Extracted Text:")
        st.text_area("Copied from your Resume:", value=extracted_text, height=150)
        st.success("You can copy the extracted text and paste it into the boxes above!")

if st.button("Check Match Score"):
    if jd_text and resume_text:
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([jd_text, resume_text])
        
        match_score = cosine_similarity(vectors[0], vectors[1])[0][0]
        match_percentage = round(match_score * 100, 2)
        
        st.success(f"✅ Resume Match Score: {match_percentage}%")
    else:
        st.warning("⚠️ Please provide text in both Job Description and Resume boxes!")