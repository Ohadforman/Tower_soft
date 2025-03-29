import os
import json
from PyPDF2 import PdfMerger
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from recent_csv import get_most_recent_csv  # Import the function

# Load directory configuration from JSON
CONFIG_FILE = "directory_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as file:
        config = json.load(file)
else:
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

# Ask user for Fiber Name and Tower Operator
fiber_name = input("Enter Fiber Name: ")
tower_operator = input("Enter Tower Operator Name: ")
drawing_data = input("Enter Drawing Data: ")

# Logo path (Ensure this image exists)
logo_path = "icap.png"

def create_general_info_page(output_filename):
    """
    Creates a first standalone title page with only the logo, fiber name, and tower operator.
    """
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter

    # Set background logo if available
    if os.path.exists(logo_path):
        logo = ImageReader(logo_path)
        c.drawImage(logo, 100, height - 200, width=400, height=100, mask='auto')  # Adjust placement

    # Fiber and Operator Details
    c.setFont("Helvetica", 20)
    c.drawCentredString(width / 2, height - 300, f"Fiber Name: {fiber_name}")
    c.drawCentredString(width / 2, height - 340, f"Tower Operator: {tower_operator}")
    c.drawCentredString(width / 2, height - 380, f"Drawing Date: {drawing_data}")

    c.save()

def create_section_title_page(section_title, output_filename):
    """
    Creates a single-page PDF with only the section title.
    """
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 250, section_title)

    c.save()

def merge_pdfs_with_titles(pdf_files, section_titles, output_filename):
    """
    Merges multiple PDFs with a first general page + section title pages before each report.
    """
    if len(pdf_files) != len(section_titles):
        raise ValueError("Each PDF must have a corresponding section title.")

    merger = PdfMerger()

    # **First page with general information**
    general_info_page = "general_info.pdf"
    create_general_info_page(general_info_page)
    merger.append(general_info_page)
    os.remove(general_info_page)

    for pdf, section_title in zip(pdf_files, section_titles):
        section_title_page = f"title_{section_title.replace(' ', '_')}.pdf"
        create_section_title_page(section_title, section_title_page)

        # Add section title page and actual PDF
        merger.append(section_title_page)
        merger.append(pdf)

        # Remove the temporary section title page
        os.remove(section_title_page)

    # Save merged PDF
    merger.write(output_filename)
    merger.close()

    print(f"✅ Merged PDF saved as: {output_filename}")

# **Find the most recent CSV and generate a folder for the merged PDF output**
folder_path = config["logs_directory"]  # Use path from JSON
recent_csv = get_most_recent_csv()  # Get most recent CSV filename
folder_out_name = os.path.splitext(recent_csv)[0]  # Remove extension
output_directory = os.path.join(config["output_folders"], folder_out_name)  # Save in a unique folder

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# File paths for PDFs (now located inside the respective output folder)
fiber_report_pdf = os.path.join(output_directory, config["fiber_report_pdf"])
coating_report_pdf = os.path.join(output_directory, config["coating_report_pdf"])
correlation_report_pdf = os.path.join(output_directory, config["correlation_report_pdf"])

# Define sections
pdf_files = [fiber_report_pdf, coating_report_pdf, correlation_report_pdf]
section_titles = [
    "Section A - Fiber Good Zones for T&M",
    "Section B - Coating Report",
    "Section C - Tower Performance Analysis"
]

# **Merged PDF should be saved in the same folder**
merged_pdf_filename = os.path.join(output_directory, config["final_merged_pdf"])

# Run the merging function
merge_pdfs_with_titles(pdf_files, section_titles, merged_pdf_filename)
