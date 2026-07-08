"""arXiv paper source."""

import logging

import httpx
from xml.etree import ElementTree as ET

from app.paper_sources.base import PaperSource, RawPaper

logger = logging.getLogger(__name__)

NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivSource(PaperSource):
    name = "arxiv"
    base_url = "https://export.arxiv.org/api/query"

    async def search(self, query: str, limit: int = 15) -> list[RawPaper]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        }
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(self.base_url, params=params)
                if resp.status_code != 200:
                    logger.error("arXiv error %d", resp.status_code)
                    return []
                xml_text = resp.text
        except Exception as e:
            logger.error("arXiv request failed: %s", e)
            return []

        papers: list[RawPaper] = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", NS):
                arxiv_url = entry.find("atom:id", NS)
                arxiv_full_id = (arxiv_url.text or "").rsplit("/", 1)[-1] if arxiv_url is not None else ""

                title_el = entry.find("atom:title", NS)
                title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""

                summary_el = entry.find("atom:summary", NS)
                abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""

                authors = []
                for author in entry.findall("atom:author", NS):
                    name_el = author.find("atom:name", NS)
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text.strip())

                published = entry.find("atom:published", NS)
                year = None
                if published is not None and published.text:
                    year = int(published.text[:4])

                link = ""
                pdf_link = ""
                for l in entry.findall("atom:link", NS):
                    if l.get("title") == "pdf":
                        pdf_link = l.get("href", "")
                    elif l.get("type") == "text/html":
                        link = l.get("href", "")

                papers.append(RawPaper(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    year=year,
                    venue="arXiv",
                    arxiv_id=arxiv_full_id,
                    url=link or f"https://arxiv.org/abs/{arxiv_full_id}",
                    pdf_url=pdf_link,
                    citation_count=0,
                    source=self.name,
                    raw_data={"arxiv_id": arxiv_full_id},
                ))
        except Exception as e:
            logger.error("arXiv XML parse failed: %s", e)

        return papers
