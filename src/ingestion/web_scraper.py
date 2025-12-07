import time
import json
import logging
import re
import requests
from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebScraper:
    """
    A scraper optimized for extracting technical documentation.
    Paths are managed relative to the project root.
    """

    def __init__(
        self, raw_dir: Optional[str] = None, processed_dir: Optional[str] = None
    ):
        self.project_root = Path(__file__).resolve().parents[3]

        if raw_dir:
            self.raw_dir = Path(raw_dir)
        else:
            self.raw_dir = self.project_root / "data" / "raw"

        if processed_dir:
            self.processed_dir = Path(processed_dir)
        else:
            self.processed_dir = self.project_root / "data" / "processed" / "web_data"

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_html(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def clean_element(self, element: Tag) -> None:
        """Removes unwanted tags like images, iframes, and separators."""
        # Remove standard unwanted tags
        for tag_name in ["img", "iframe", "hr"]:
            for tag in element.find_all(tag_name):
                tag.decompose()

        # Remove generic spacers (often used in Tailwind)
        for div in element.find_all("div", class_="mt-32"):
            div.decompose()

        for div in element.find_all("div", class_="lg:hidden mt-48"):
            div.decompose()

        # Unwrap links to keep text but remove anchor behavior
        for a_tag in element.find_all("a"):
            a_tag.unwrap()

    def _extract_outputs_for_block(self, start_node: Tag) -> List[Tag]:
        """
        Looks ahead from the code block to find associated outputs.
        Returns a list of output nodes to be processed and removed.
        """
        collected_outputs = []
        found_output_label = False

        # Look at siblings immediately following the code block
        for sibling in start_node.find_next_siblings():
            if isinstance(sibling, str) and not sibling.strip():
                continue

            # Skip spacer divs (common in Tailwind layouts as seen in screenshot)
            if sibling.name == "div" and "mt-32" in sibling.get("class", []):
                continue

            # Check for the "Output:" label
            if not found_output_label:
                if sibling.name == "p" and "Output:" in sibling.get_text():
                    found_output_label = True
                    collected_outputs.append(sibling)  # Add label to removal list
                    continue
                else:
                    break

            # If we found the label, collect the snippet divs
            if found_output_label:
                if sibling.name == "div" and "snippet" in sibling.get("class", []):
                    collected_outputs.append(sibling)
                else:
                    # End of output stream
                    break

        return collected_outputs

    def _process_code_blocks(self, prose: Tag) -> None:
        """
        Finds code blocks, looks for their outputs, and merges them into a single
        text representation suitable for embedding.
        """
        # Find all code blocks marked by the specific attribute
        code_blocks = prose.find_all(
            "div", attrs={"data-rehype-pretty-code-fragment": True}
        )

        for block in code_blocks:
            # 1. Extract the source code
            code_text = block.get_text(strip=True)

            # 2. Find associated outputs (if any)
            output_nodes = self._extract_outputs_for_block(block)

            output_text_parts = []
            for node in output_nodes:
                # We skip the label node text itself, we just want the snippets
                if "snippet" in node.get("class", []):
                    output_text_parts.append(node.get_text(strip=True))
                node.decompose()

            # 3. Format the combined text
            combined_text = f"\n```\n{code_text}\n```\n"

            if output_text_parts:
                combined_output = "\n".join(output_text_parts)
                combined_text += f"Output:\n```\n{combined_output}\n```\n"

            # 4. Replace the original HTML block with our formatted text
            block.replace_with(combined_text)

    def _process_latex(self, prose: Tag) -> None:
        """
        Finds LaTeX spans (katex-display) and wraps them in custom markers.
        """
        # Find all spans with class 'katex-display'
        latex_nodes = prose.find_all("span", class_="katex-display")

        for node in latex_nodes:
            # Extract the LaTeX content
            latex_text = node.get_text(strip=True)

            # Create the formatted string with markers
            formatted_text = f" [LATEX_START] {latex_text} [LATEX_END] "

            # Replace the node with the formatted text
            node.replace_with(formatted_text)

    def _process_headers(self, prose: Tag) -> None:
        """
        Preserves HTML headers (h1-h6) by replacing the tag object with
        its text representation (e.g. <h1>Title</h1>) so it survives get_text().
        """
        for i in range(1, 7):
            tag_name = f"h{i}"
            headers = prose.find_all(tag_name)
            for header in headers:
                header_text = header.get_text(strip=True)
                if header_text:
                    # Replace the actual tag with a string containing the tag
                    preserved_header = f"\n<{tag_name}>{header_text}</{tag_name}>\n"
                    header.replace_with(preserved_header)

    def parse_content(self, html: str, url: str) -> Optional[Dict]:
        soup = BeautifulSoup(html, "html.parser")

        prose = soup.find("div", class_="prose")
        if not prose:
            logger.warning(f"No <div class='prose'> found in {url}")
            return None

        h1_tag = prose.find("h1")
        topic = (
            h1_tag.get("id") if h1_tag and h1_tag.has_attr("id") else "unknown_topic"
        )
        if topic == "unknown_topic" and h1_tag:
            topic = re.sub(
                r"[^a-z0-9]+", "-", h1_tag.get_text(strip=True).lower()
            ).strip("-")

        has_code = bool(
            prose.find("div", attrs={"data-rehype-pretty-code-fragment": True})
        )
        has_latex = bool(prose.find("span", class_="katex-display"))

        # Process Code Blocks
        self._process_code_blocks(prose)

        # Process LaTeX Blocks
        self._process_latex(prose)

        # Process Headers
        self._process_headers(prose)

        # General cleaning
        self.clean_element(prose)

        content_text = prose.get_text(separator="\n", strip=True)

        return {
            "metadata": {
                "url": url,
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "has_code": has_code,
                "has_latex": has_latex,
            },
            "content": content_text,
        }

    def save_data(self, data: Dict):
        """Saves a single data object to a separate JSON file."""
        topic = data["metadata"]["topic"]
        safe_filename = re.sub(r"[^a-z0-9\-_]", "", topic)
        if not safe_filename:
            safe_filename = f"page_{int(time.time())}"

        file_path = self.processed_dir / f"{safe_filename}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved: {file_path}")

    def run(self, urls_file: str = "urls.txt"):
        urls_path = self.raw_dir / urls_file

        if not urls_path.exists():
            logger.error(f"URLs file not found: {urls_path}")
            return

        with open(urls_path, "r") as f:
            urls = [line.strip() for line in f if line.strip()]

        logger.info(f"Found {len(urls)} URLs to process.")

        for i, url in enumerate(urls):
            logger.info(f"Processing ({i+1}/{len(urls)}): {url}")

            html = self.fetch_html(url)
            if html:
                data = self.parse_content(html, url)
                if data:
                    self.save_data(data)

            if i < len(urls) - 1:
                logger.info("Waiting 10 seconds...")
                time.sleep(10)


if __name__ == "__main__":
    scraper = WebScraper()
    scraper.run("urls.txt")
