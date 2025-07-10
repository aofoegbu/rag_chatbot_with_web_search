#!/usr/bin/env python3
"""
Create a sample image with text for testing OCR functionality
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_image():
    # Create a white background image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fall back to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
    except:
        try:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        except:
            font = None
            title_font = None
    
    # Add text content
    y_position = 50
    
    # Title
    title_text = "Ogelo RAG Chat Assistant"
    if title_font:
        draw.text((50, y_position), title_text, fill='black', font=title_font)
    else:
        draw.text((50, y_position), title_text, fill='black')
    y_position += 100
    
    # Content
    content_lines = [
        "This is a sample image with text content.",
        "The OCR system should be able to extract this text",
        "and add it to the knowledge base for searching.",
        "",
        "Key Features:",
        "• Document processing with OCR support",
        "• Web content extraction from URLs", 
        "• Multi-format file support (PDF, CSV, images)",
        "• PostgreSQL and SQLite database backends",
        "• Conversation history integration",
        "",
        "This text demonstrates the OCR capabilities",
        "of the Ogelo RAG Chat Assistant system."
    ]
    
    for line in content_lines:
        if font:
            draw.text((50, y_position), line, fill='black', font=font)
        else:
            draw.text((50, y_position), line, fill='black')
        y_position += 35
    
    # Save the image
    image.save('sample_ocr_test.png')
    print("Sample OCR test image created: sample_ocr_test.png")

if __name__ == "__main__":
    create_sample_image()