import os
import time
import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag, NavigableString


class Config:
    """Central configuration for file paths and scraper settings."""

    RAW_URL_FILE = "data/raw/api_url.txt"
    OUTPUT_DIR = "data/processed/qiskit_api"
    PY_FILES_DIR = "data/raw/py_files"
    DELAY_SECONDS = 10
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("QiskitScraper")


class GitHubHandler:
    """Handles logic related to GitHub URLs and file downloading."""

    def __init__(self, download_dir: str):
        self.download_dir = download_dir
        # Set to track URLs processed in this session to avoid redundant requests
        self.visited_urls = set()
        os.makedirs(self.download_dir, exist_ok=True)

    def is_github_source_link(self, tag: Tag) -> bool:
        """Checks if an <a> tag is a 'view source code' link pointing to GitHub."""
        if not tag.name == "a":
            return False

        title = tag.get("title", "")
        href = tag.get("href", "")

        if "view source code" in title and "github.com" in href:
            return True
        return False

    def convert_blob_to_raw(self, url: str) -> str:
        """Converts a GitHub blob or tree URL to a raw content URL."""
        # Remove line fragments (e.g., #L108-L203)
        url = url.split("#")[0]

        # Replace domain
        url = url.replace("github.com", "raw.githubusercontent.com")
        url = url.replace("/blob/", "/").replace("/tree/", "/")

        return url

    def download_file(self, url: str) -> Optional[str]:
        """
        Downloads the file from GitHub.
        Prevents duplicates by checking both runtime history and file system.
        """
        try:
            raw_url = self.convert_blob_to_raw(url)
            filename = raw_url.split("/")[-1]
            save_path = os.path.join(self.download_dir, filename)

            # 1. Check runtime session (fastest check)
            if raw_url in self.visited_urls:
                return filename

            # 2. Check file system (persistence check)
            if os.path.exists(save_path):
                logger.info(
                    f"File already exists on disk (Skipping download): {filename}"
                )
                self.visited_urls.add(raw_url)  # Mark as visited
                return filename

            # 3. Download if new
            logger.info(f"Downloading new GitHub file: {filename}")
            response = requests.get(raw_url)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                f.write(response.content)

            self.visited_urls.add(raw_url)
            return filename

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None


class ContentParser:
    """
    Parses HTML content based on inclusion/exclusion rules.
    """

    def __init__(self, github_handler: GitHubHandler):
        self.github_handler = github_handler
        self.has_latex = False
        self.has_code = False
        # Use a Set instead of List to prevent duplicate filenames in metadata
        self.downloaded_files = set()

    def reset_metadata(self):
        self.has_latex = False
        self.has_code = False
        self.downloaded_files = set()

    def clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def process_node(self, node) -> str:
        """
        Recursive function to traverse and process nodes.
        """
        if isinstance(node, NavigableString):
            return str(node)

        if not isinstance(node, Tag):
            return ""

        # 1. Exclude specific div class: <div class="lg:hidden mt-48">
        classes = node.get("class", [])
        if node.name == "div" and "lg:hidden" in classes and "mt-48" in classes:
            return ""

        # 2. Exclude unwanted tags
        if node.name in ["img", "iframe", "hr", "script", "style"]:
            return ""

        # 1. Python Code Blocks: <div data-rehype-pretty-code-fragment>
        if node.name == "div" and node.has_attr("data-rehype-pretty-code-fragment"):
            self.has_code = True
            code_text = node.get_text()
            return f"\n```python\n{code_text}\n```\n"

        # 2. LaTeX: <span class="katex-display">
        if node.name == "span" and "katex-display" in classes:
            self.has_latex = True
            return f" [LATEX_START] {node.get_text()} [LATEX_END] "

        # 3. GitHub Source Links (Download logic here)
        if self.github_handler.is_github_source_link(node):
            file_url = node.get("href")
            filename = self.github_handler.download_file(file_url)
            if filename:
                # Add to set (handles duplicates automatically)
                self.downloaded_files.add(filename)
            return ""

        # 4. Links with specific titles (e.g., "(in Python v3.14)")
        if node.name == "a":
            title = node.get("title", "")
            if "(in Python" in title:
                # Pass through to process children (<em> tags)
                pass
            elif node.name == "a":
                # General rule: remove <a> tags but keep content if needed.
                # Assuming logic: Strip tag, keep text.
                pass

        # 5. Spans (General): Skip tag, process children
        if node.name == "span" and "katex-display" not in classes:
            pass

        content_parts = []
        for child in node.children:
            content_parts.append(self.process_node(child))

        inner_content = "".join(content_parts)

        if node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            return f"\n\n<{node.name}>{inner_content}</{node.name}>\n"

        if node.name == "p":
            return f"\n{inner_content}\n"

        if node.name == "ul":
            return f"\n{inner_content}\n"

        if node.name == "li":
            return f"- {inner_content}\n"

        return inner_content

    def extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extracts the main title.
        Instead of relying on a specific ID (which changes), we look for the first <h1> tag.
        Priority:
        1. <h1> inside the 'prose' content div (most accurate).
        2. First <h1> anywhere on the page.
        3. <title> tag.
        """
        # 1. Try to find h1 within the main content area first
        prose = soup.find("div", class_="prose")
        if prose:
            h1 = prose.find("h1")
            if h1:
                return self.clean_text(h1.get_text())

        # 2. Fallback: Any h1 tag on the page
        h1_generic = soup.find("h1")
        if h1_generic:
            return self.clean_text(h1_generic.get_text())

        # 3. Fallback: Page title
        if soup.title:
            return self.clean_text(soup.title.get_text())

        return "No_Title"


class QiskitScraper:
    """
    Main manager class for the scraping process.
    """

    def __init__(self):
        self.github_handler = GitHubHandler(Config.PY_FILES_DIR)
        self.parser = ContentParser(self.github_handler)

        # Ensure directories exist
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    def load_urls(self) -> List[str]:
        if not os.path.exists(Config.RAW_URL_FILE):
            logger.error(f"URL file not found: {Config.RAW_URL_FILE}")
            return []

        with open(Config.RAW_URL_FILE, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls

    def fetch_page(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers={"User-Agent": Config.USER_AGENT})
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def start(self):
        logger.info("Starting Qiskit Scraper...")
        urls = self.load_urls()
        logger.info(f"Found {len(urls)} URLs to process.")

        for i, url in enumerate(urls):
            logger.info(f"Processing ({i+1}/{len(urls)}): {url}")

            html = self.fetch_page(url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            prose_div = soup.find("div", class_="prose")

            if not prose_div:
                logger.warning(f"No <div class='prose'> found in {url}")
                continue

            self.parser.reset_metadata()

            # Pass soup to extract title (it searches globally or in prose)
            title = self.parser.extract_title(soup)

            content = self.parser.process_node(prose_div)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()

            record = {
                "url": url,
                "title": title,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "has_code": self.parser.has_code,
                    "has_latex": self.parser.has_latex,
                    "downloaded_py_files": list(self.parser.downloaded_files),
                },
                "content": content,
            }

            self.save_single_record(record)

            logger.info(f"Waiting {Config.DELAY_SECONDS} seconds...")
            time.sleep(Config.DELAY_SECONDS)

        logger.info("Scraping completed.")

    def save_single_record(self, record: Dict):
        """Saves a single record to a JSON file named after the title or URL slug."""
        title = record.get("title", "No_Title")

        # Sanitize filename: keep only alphanumeric, space, hyphen, underscore
        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")

        # Fallback: If title is missing or generic, derive filename from URL
        if not safe_title or safe_title == "No_Title":
            url_path = urlparse(record.get("url", "")).path
            # Get the last segment of the URL (e.g., 'QuantumVolume' from '.../QuantumVolume')
            safe_title = url_path.strip("/").split("/")[-1]

            if not safe_title:
                safe_title = f"untitled_{int(time.time())}"

        filename = f"{safe_title}.json"
        file_path = os.path.join(Config.OUTPUT_DIR, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=4)
            logger.info(f"Record saved to {file_path}")
        except IOError as e:
            logger.error(f"Failed to save JSON for {title}: {e}")


if __name__ == "__main__":
    # Setup dummy data for testing if not exists
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
        if not os.path.exists(Config.RAW_URL_FILE):
            with open(Config.RAW_URL_FILE, "w") as f:
                f.write(
                    "https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.QuantumVolume"
                )

    scraper = QiskitScraper()
    scraper.start()
