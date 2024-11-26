import PyPDF2
import fitz  # PyMuPDF
from utils.logger import get_logger

logger = get_logger(__name__)

def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un fichier PDF en utilisant PyPDF2 puis PyMuPDF en cas d'échec.
    Args:
        pdf_path (str): Chemin du fichier PDF.
    Returns:
        str: Texte extrait.
    """
    text = ""

    # Essayer avec PyPDF2
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        logger.info(f"Text extracted using PyPDF2: {pdf_path}")
        return text
    except Exception as e:
        logger.warning(f"PyPDF2 failed for {pdf_path}: {e}")

    # Essayer avec PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        logger.info(f"Text extracted using PyMuPDF: {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"PyMuPDF failed for {pdf_path}: {e}")

    raise RuntimeError(f"Failed to extract text from {pdf_path} using both PyPDF2 and PyMuPDF.")
