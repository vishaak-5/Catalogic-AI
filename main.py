import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = FastAPI(title="Catalogic AI - MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Catalogic AI Engine is running!"}

@app.post("/extract-specs/")
async def extract_specs(
    file: UploadFile = File(...), 
    part_number: str = Form(...) # We ask for the part number we want to find!
):
    # 1. Save the uploaded PDF temporarily to disk
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Extract text from the PDF and split it into readable chunks
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # 3. Create local embeddings and load them into ChromaDB
        # This downloads a small, free model to your machine to convert text to vectors
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)

        # 4. Search the document for the specific part number
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(part_number)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 5. Send the found context to the LLM to extract structured data
        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key="YOUR_GROQ_API_KEY_HERE")
        
        prompt = PromptTemplate.from_template(
            """You are an expert industrial engineer. Look at the following technical manual context and extract the specifications for Part Number: {part_number}.
            
            Context from manual:
            {context}
            
            Return ONLY a JSON object containing attributes like Material, Dimensions, Weight, and Voltage if they exist in the text. If you can't find them, put "Not specified".
            """
        )
        
        chain = prompt | llm
        response = chain.invoke({"part_number": part_number, "context": context})

        return {
            "status": "success",
            "part_number": part_number,
            "raw_ai_output": response.content
        }

    finally:
        # Clean up the temporary file after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)