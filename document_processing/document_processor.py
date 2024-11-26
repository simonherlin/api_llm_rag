import os
from utils.pdf_processor import extract_text_from_pdf
from utils.logger import get_logger

logger = get_logger(__name__)

def process_documents(input_dir, paragraph_splitter=None):
    """
    Charge les documents (txt et pdf) et les découpe en paragraphes.
    Args:
        input_dir (str): Répertoire contenant les fichiers.
        paragraph_splitter (callable): Fonction pour découper le texte en paragraphes.
    Returns:
        dict: Dictionnaire avec le nom du fichier comme clé et une liste de paragraphes comme valeur.
    """
    logger.info(f"Loading documents from {input_dir}")
    documents = {}
    paragraph_splitter = paragraph_splitter or (lambda text: text.split("\n\n"))

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    documents[filename] = paragraph_splitter(content)
                logger.info(f"Processed TXT file: {filename}")
            except Exception as e:
                logger.error(f"Failed to process TXT file {filename}: {e}")
        elif filename.endswith(".pdf"):
            try:
                content = extract_text_from_pdf(file_path)
                documents[filename] = paragraph_splitter(content)
                logger.info(f"Processed PDF file: {filename}")
            except Exception as e:
                logger.error(f"Failed to process PDF file {filename}: {e}")

    logger.info(f"Processed {len(documents)} documents.")
    return documents
