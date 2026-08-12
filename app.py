import streamlit as st
import requests
import json

# Set the page layout
st.set_page_config(page_title="Catalogic AI", layout="centered")

st.title("🏭 Catalogic AI Dashboard")
st.subheader("B2B Product Intelligence Extractor")
st.markdown("Upload a raw manufacturer datasheet to automatically extract structured B2B attributes.")

# Create the UI layout
st.divider()
uploaded_file = st.file_uploader("1. Upload Technical Manual (PDF)", type=["pdf"])
part_number = st.text_input("2. Enter Target Part Number / Product Name")

if st.button("Generate Product Intelligence", type="primary"):
    if uploaded_file and part_number:
        with st.spinner("AI Pipeline Active: Reading vectors and extracting specs..."):
            
            # Package the file and text to send to your FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"part_number": part_number}
            
            try:
                # Call your local backend
                response = requests.post("http://127.0.0.1:8000/extract-specs/", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Extraction Complete! Confidence Score: High")
                    
                    # Parse the stringified JSON from the LLM back into a real dictionary
                    try:
                        raw_ai_dict = json.loads(result["raw_ai_output"])
                    except:
                        raw_ai_dict = {"raw_text": result["raw_ai_output"]} # Fallback if LLM didn't format perfectly
                    
                    # Display the data in a beautiful table/JSON view
                    st.subheader("Structured B2B Attributes")
                    st.json(raw_ai_dict)
                    
                else:
                    st.error(f"Backend Server Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Is your FastAPI server running on port 8000?")
    else:
        st.warning("Please upload a PDF and enter a part number to begin.")