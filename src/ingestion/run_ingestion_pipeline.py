import logging
import time
from src.ingestion.api_docs import QiskitScraper
from src.ingestion.notebook_processor import NotebookProcessor
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.web_scraper import WebScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IngestionPipeline")


def run_pipeline():
    logger.info("Starting Global Ingestion Pipeline...")
    start_time = time.time()

    # 1. API Documentation (Qiskit API)
    logger.info("=== Stage 1: API Documentation Scraper ===")
    try:
        api_scraper = QiskitScraper()
        api_scraper.start()
    except Exception as e:
        logger.error(f"Stage 1 Failed: {e}")

    # 2. Web Scraper (General Documentation)
    logger.info("=== Stage 2: Web Scraper ===")
    try:
        web_scraper = WebScraper()
        web_scraper.run("urls.txt")
    except Exception as e:
        logger.error(f"Stage 2 Failed: {e}")

    # 3. PDF Processing
    logger.info("=== Stage 3: PDF Processor ===")
    try:
        pdf_processor = PDFProcessor()
        pdf_processor.run()
    except Exception as e:
        logger.error(f"Stage 3 Failed: {e}")

    # 4. Notebook Processing
    logger.info("=== Stage 4: Notebook Processor ===")
    try:
        nb_processor = NotebookProcessor()
        nb_processor.process_directory()
    except Exception as e:
        logger.error(f"Stage 4 Failed: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Ingestion Pipeline Completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    run_pipeline()
