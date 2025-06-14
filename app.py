import streamlit as st
import os
import traceback
from dotenv import load_dotenv
import logging
from src.database.connection import get_db_connection, store_document, store_document_chunk, store_feedback
from src.embeddings.generator import generate_embeddings, chunk_text
from src.rag.retriever import retrieve_relevant_docs
from src.rag.generator import generate_response
from src.utils.document_processor import process_document
import json
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = ['OPENAI_API_KEY', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Set page config
st.set_page_config(
    page_title="Document Analysis Assistant",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Document Analysis Assistant")
st.markdown("""
This application helps you analyze and understand any document through AI-powered question answering. 
Upload your document and ask questions about its content!

### Features:
- 📝 Support for PDF, DOCX, and TXT files
- 🖼️ Image extraction and analysis
- 🤖 AI-powered document analysis
- ❓ Ask questions about your document
- 📊 View relevant source sections
""")

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

def process_and_store_document(file):
    """Process and store a single document."""
    try:
        logger.info(f"Processing document: {file.name}")
        
        # Create progress bar for overall process
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Process document
        status_text.text("📄 Reading document...")
        doc = process_document(file)
        if not doc:
            logger.error(f"Failed to process document: {file.name}")
            return False
        progress_bar.progress(20)
        
        # Step 2: Generate chunks
        status_text.text("✂️ Splitting document into chunks...")
        chunks = chunk_text(doc['content'])
        logger.info(f"Generated {len(chunks)} chunks for {file.name}")
        progress_bar.progress(40)
        
        # Step 3: Generate embeddings with batch progress
        status_text.text("🧮 Generating embeddings...")
        total_batches = (len(chunks) + 19) // 20  # Ceiling division for batch size of 20
        embeddings = []
        
        for i in range(0, len(chunks), 20):
            batch = chunks[i:i+20]
            batch_embeddings = generate_embeddings(batch)
            embeddings.extend(batch_embeddings)
            
            # Update progress
            batch_progress = min(40 + (i + len(batch)) * 40 / len(chunks), 80)
            progress_bar.progress(int(batch_progress))
            status_text.text(f"🧮 Generating embeddings... ({i + len(batch)}/{len(chunks)} chunks)")
        
        logger.info(f"Generated embeddings for {len(embeddings)} chunks")
        progress_bar.progress(80)
        
        # Step 4: Store in database
        status_text.text("💾 Storing in database...")
        chunk_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = store_document_chunk(
                doc['title'],
                chunk,
                embedding,
                doc['metadata']
            )
            chunk_ids.append(chunk_id)
            
            # Update progress
            db_progress = 80 + (i + 1) * 20 / len(chunks)
            progress_bar.progress(int(db_progress))
            status_text.text(f"💾 Storing in database... ({i + 1}/{len(chunks)} chunks)")
        
        # Update session state
        st.session_state.processed_files[file.name] = {
            'status': 'success',
            'chunks': len(chunks),
            'chunk_ids': chunk_ids,
            'metadata': doc['metadata']
        }
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        logger.info(f"Successfully processed and stored document: {file.name}")
        return True
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        logger.error(traceback.format_exc())
        st.session_state.processed_files[file.name] = {
            'status': 'error',
            'error': str(e)
        }
        return False

def main():
    # File upload section
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt'])

    if uploaded_file is not None:
        try:
            # Process the document using the helper function
            success = process_and_store_document(uploaded_file)
            
            if success:
                # Get document info from session state
                doc_info = st.session_state.processed_files[uploaded_file.name]
                
                # Display success message
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                
                # Display image information if available
                if 'metadata' in doc_info and 'image_count' in doc_info['metadata']:
                    st.subheader("📸 Extracted Images")
                    for i in range(doc_info['metadata']['image_count']):
                        with st.expander(f"Image {i+1}"):
                            if 'image_descriptions' in doc_info['metadata']:
                                st.write(doc_info['metadata']['image_descriptions'][i])
                
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            logger.error(f"Error in document processing: {str(e)}")
            logger.error(traceback.format_exc())

    # Question answering section
    st.header("Ask Questions About Your Document")
    question = st.text_input("Enter your question about the document:")

    if question and uploaded_file is not None:
        with st.spinner("Analyzing your question..."):
            try:
                # Get the current document title
                current_doc_title = uploaded_file.name
                
                # Log the search attempt
                logger.info(f"Searching for question: '{question}' in document: '{current_doc_title}'")
                
                # Retrieve relevant chunks from the current document
                relevant_chunks = retrieve_relevant_docs(question, current_doc_title)
                
                if not relevant_chunks:
                    st.warning("No relevant information found in the current document.")
                    logger.warning(f"No chunks found for document: {current_doc_title}")
                else:
                    # Generate answer using the retrieved chunks
                    answer = generate_response(question, relevant_chunks)
                    
                    # Display the answer
                    st.subheader("Answer")
                    st.write(answer)
                    
                    # Display source sections
                    st.subheader("Source Sections")
                    for chunk in relevant_chunks:
                        with st.expander(f"Section from {chunk['title']}"):
                            st.write(chunk['content'])
                            
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
                logger.error(f"Error in question answering: {str(e)}")
                logger.error(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and AI</p>
        <p>Upload any document and start asking questions!</p>
    </div>
    """, unsafe_allow_html=True)

    # Show document status
    if uploaded_file and uploaded_file.name in st.session_state.processed_files:
        with st.sidebar:
            st.subheader("📊 Current Document Status")
            file_info = st.session_state.processed_files[uploaded_file.name]
            if file_info['status'] == 'success':
                st.success(f"✅ {uploaded_file.name}")
                st.write(f"📑 Chunks created: {file_info['chunks']}")
                if 'metadata' in file_info:
                    st.write("📋 Metadata:")
                    st.json(file_info['metadata'])
            else:
                st.error(f"❌ {uploaded_file.name}")
                st.error(f"Error: {file_info['error']}")

    # Display processing status
    if st.session_state.processed_files:
        with st.sidebar:
            st.subheader("📊 Processing Status")
            for filename, info in st.session_state.processed_files.items():
                if info['status'] == 'success':
                    st.success(f"✅ {filename}")
                    st.write(f"📑 Chunks: {info['chunks']}")
                else:
                    st.error(f"❌ {filename}")
                    st.error(f"Error: {info['error']}")

    # How to Use section in bottom right
    with st.sidebar:
        st.markdown("---")
        with st.expander("ℹ️ How to Use", expanded=False):
            st.markdown("""
            ### Quick Guide
            
            **1. Upload Document**
            - Click 'Browse files' to upload your legal document
            - Supported formats: PDF, DOCX, TXT
            - Wait for processing to complete
            
            **2. Ask Questions**
            - Type your question in the text input
            - Be specific about what you're looking for
            - Example questions:
              - "What are the key terms in section 3?"
              - "Explain the termination clause"
              - "What are the payment terms?"
              - "Summarize the liability provisions"
              - "What are the dispute resolution procedures?"
              - "List all obligations of Party A"
              - "What are the confidentiality requirements?"
              - "Explain the force majeure clause"
            
            **3. View Results**
            - Read the AI-generated answer
            - Check source documents for verification
            - Expand "View Source Documents" for context
            
            **Tips:**
            - Keep questions clear and specific
            - Use legal terminology when possible
            - Check source documents for accuracy
            - One document at a time for best results
            - Use section numbers when referencing specific parts
            - Ask follow-up questions for clarification
            
            **Common Questions:**
            Q: How accurate are the answers?
            A: Answers are based on the uploaded document content. Always verify with source documents.
            
            Q: Can I upload multiple documents?
            A: Currently, the system works best with one document at a time for focused analysis.
            
            Q: What types of legal documents are supported?
            A: The system works with contracts, agreements, legal memos, and other legal documents in PDF, DOCX, or TXT format.
            
            Q: How long does processing take?
            A: Processing time depends on document size and complexity. Most documents are processed within a few seconds.
            
            **Need Help?**
            - Check the document status in sidebar
            - Look for error messages in red
            - Ensure your document is properly formatted
            - Contact support if issues persist
            """)
            
            # Add feedback section
            st.markdown("---")
            st.markdown("### Feedback")
            
            if not st.session_state.feedback_submitted:
                # Feedback type selection
                feedback_type = st.selectbox(
                    "Feedback Type",
                    ["Suggestion", "Bug Report", "Question", "Other"]
                )
                
                # Rating
                rating = st.slider("Rate your experience", 1, 5, 3)
                
                # Feedback content
                feedback = st.text_area("Help us improve! Share your feedback:", height=100)
                
                # Optional email
                email = st.text_input("Email (optional)")
                
                if st.button("Submit Feedback"):
                    if feedback:
                        try:
                            store_feedback(
                                feedback_type=feedback_type,
                                content=feedback,
                                user_email=email if email else None,
                                rating=rating
                            )
                            st.session_state.feedback_submitted = True
                            st.success("Thank you for your feedback!")
                        except Exception as e:
                            logger.error(f"Error storing feedback: {str(e)}")
                            st.error("Failed to submit feedback. Please try again.")
                    else:
                        st.warning("Please provide feedback before submitting.")

if __name__ == "__main__":
    main() 