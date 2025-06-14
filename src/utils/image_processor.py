import os
import logging
from PIL import Image
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from docx import Document
import io
import tempfile

logger = logging.getLogger(__name__)

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def extract_images_from_pdf(pdf_file):
    """
    Extract images from a PDF file.
    
    Args:
        pdf_file: PDF file object or path
        
    Returns:
        List of extracted images and their text content
    """
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_file)
        results = []
        
        for i, image in enumerate(images):
            try:
                # Convert PIL image to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Process image
                processed_image, text = process_image(opencv_image)
                
                results.append({
                    'page': i + 1,
                    'image': processed_image,
                    'text': text
                })
                
                logger.info(f"Successfully processed page {i+1}")
            except Exception as page_error:
                logger.error(f"Error processing page {i+1}: {str(page_error)}")
                continue
            
        return results
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {str(e)}")
        raise

def extract_images_from_docx(docx_file):
    """
    Extract images from a DOCX file.
    
    Args:
        docx_file: DOCX file object or path
        
    Returns:
        List of extracted images and their text content
    """
    try:
        doc = Document(docx_file)
        results = []
        
        for i, rel in enumerate(doc.part.rels.values()):
            if "image" in rel.target_ref:
                try:
                    # Get image data
                    image_data = rel.target_part.blob
                    
                    # Convert to OpenCV format
                    nparr = np.frombuffer(image_data, np.uint8)
                    opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Process image
                    processed_image, text = process_image(opencv_image)
                    
                    results.append({
                        'image_number': i + 1,
                        'image': processed_image,
                        'text': text
                    })
                    
                    logger.info(f"Successfully processed image {i+1}")
                except Exception as img_error:
                    logger.error(f"Error processing image {i+1}: {str(img_error)}")
                    continue
                
        return results
    except Exception as e:
        logger.error(f"Error extracting images from DOCX: {str(e)}")
        raise

def process_image(image):
    """
    Process an image to enhance text extraction.
    
    Args:
        image: OpenCV image
        
    Returns:
        Tuple of (processed image, extracted text)
    """
    try:
        # Check if image is already grayscale
        if len(image.shape) == 2:
            gray = image
        else:
            # Convert to grayscale if it's a color image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply dilation to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)
        
        # Extract text using pytesseract with custom configuration
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(gray, config=custom_config)
        
        logger.info(f"Successfully extracted text from image")
        return gray, text
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def save_processed_images(images, output_dir):
    """
    Save processed images to disk.
    
    Args:
        images: List of processed images
        output_dir: Directory to save images
        
    Returns:
        List of saved image paths
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, img_data in enumerate(images):
            try:
                # Convert OpenCV image to PIL Image
                pil_image = Image.fromarray(img_data['image'])
                
                # Save image
                image_path = os.path.join(output_dir, f'processed_image_{i+1}.png')
                pil_image.save(image_path)
                saved_paths.append(image_path)
                
                logger.info(f"Successfully saved image {i+1}")
            except Exception as save_error:
                logger.error(f"Error saving image {i+1}: {str(save_error)}")
                continue
            
        return saved_paths
    except Exception as e:
        logger.error(f"Error saving processed images: {str(e)}")
        raise

def get_image_description(image):
    """
    Get a description of the image content.
    
    Args:
        image: OpenCV image
        
    Returns:
        String description of the image
    """
    try:
        # Extract text from image
        _, text = process_image(image)
        
        # Basic image analysis
        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        
        description = f"Image dimensions: {width}x{height}, Channels: {channels}\n"
        if text.strip():
            description += f"Extracted text: {text.strip()}"
        else:
            description += "No text detected in image"
            
        logger.info(f"Successfully generated image description")
        return description
    except Exception as e:
        logger.error(f"Error getting image description: {str(e)}")
        raise 