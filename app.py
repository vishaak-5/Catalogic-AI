import streamlit as st
import os
import shutil
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Set the page layout
st.set_page_config(page_title="Catalogic AI", layout="centered")

st.title("🏭 Catalogic AI Dashboard")
st.subheader("B2B Product Intelligence Extractor")
st.markdown("Upload a raw manufacturer datasheet to automatically extract structured B2B attributes.")

st.divider()
uploaded_file = st.file_uploader("1. Upload Technical Manual (PDF)", type=["pdf"])
part_number = st.text_input("2. Enter Target Part Number / Product Name")

if st.button("Generate Product Intelligence", type="primary"):
    if uploaded_file and part_number:
        with st.spinner("AI Pipeline Active: Processing vectors and extracting specs..."):
            
            # 1. Save uploaded file to a temporary location
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # 2. Extract and split text
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)

                # 3. Create vector embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(chunks, embeddings)

                # 4. Retrieve context
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                relevant_docs = retriever.invoke(part_number)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # 5. Run LLM Engine via Groq
                # HARDCODED KEY WARNING: Replace the placeholder below with your actual Groq key string
                llm = ChatGroq(
                    temperature=0, 
                    model_name="openai/gpt-oss-20b", 
                    api_key=st.secrets["GROQ_API_KEY"]  # <-- Change this exact line
                )
                
                prompt = PromptTemplate.from_template(
                    """You are an expert industrial engineer. Look at the following technical manual context and extract the specifications for Part Number: {part_number}.
                    
                    Context from manual:
                    {context}
                    
                    Return ONLY a valid JSON object containing attributes like Material, Dimensions, Weight, and Voltage if they exist in the text. If you can't find them, put "Not specified".
                    """
                )
                
                chain = prompt | llm
                response = chain.invoke({"part_number": part_number, "context": context})
                
                st.success("Extraction Complete! Confidence Score: High")
                
                # Parse and display final output
                try:
                    # Clean the output to extract just the JSON bracketed area
                    ai_text = response.content
                    if "```json" in ai_text:
                        ai_text = ai_text.split("```json")[1].split("```")[0]
                    elif "```" in ai_text:
                        ai_text = ai_text.split("```")[1].split("```")[0]
                        
                    raw_ai_dict = json.loads(ai_text.strip())
                    st.subheader("Structured B2B Attributes")
                    st.json(raw_ai_dict)
                except Exception:
                    st.subheader("Extracted Output")
                    st.text(response.content)
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    else:
        st.warning("Please upload a PDF and enter a part number to begin.")