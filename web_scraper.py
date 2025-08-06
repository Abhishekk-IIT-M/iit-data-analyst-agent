import trafilatura
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional
import re

logger = logging.getLogger(__name__)

def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Some common website to crawl information from:
    MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text if text else ""
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return ""

def scrape_table_data(url: str) -> List[Dict]:
    """
    Scrape table data from a webpage
    Returns structured data that can be converted to DataFrame
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        all_table_data = []
        
        for i, table in enumerate(tables):
            table_data = []
            
            # Get headers
            headers = []
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text().strip())
            
            # Get data rows
            rows = table.find_all('tr')[1:] if header_row else table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = {}
                    for j, cell in enumerate(cells):
                        header = headers[j] if j < len(headers) else f'Column_{j}'
                        row_data[header] = cell.get_text().strip()
                    table_data.append(row_data)
            
            if table_data:
                all_table_data.extend(table_data)
        
        return all_table_data
        
    except Exception as e:
        logger.error(f"Error scraping table data from {url}: {str(e)}")
        return []

def scrape_wikipedia_data(topic: str) -> Dict[str, str]:
    """
    Scrape Wikipedia data for a given topic
    Returns structured information including summary, sections, etc.
    """
    try:
        # Format Wikipedia URL
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            return {}
        
        # Get summary (first paragraph)
        summary_p = content.find('p')
        summary = summary_p.get_text().strip() if summary_p else ""
        
        # Get section headings and content
        sections = {}
        headings = content.find_all(['h2', 'h3', 'h4'])
        
        for heading in headings:
            section_title = heading.get_text().strip()
            section_content = []
            
            # Get content until next heading
            current = heading.find_next_sibling()
            while current and current.name not in ['h2', 'h3', 'h4']:
                if current.name == 'p':
                    section_content.append(current.get_text().strip())
                current = current.find_next_sibling()
            
            if section_content:
                sections[section_title] = ' '.join(section_content)
        
        return {
            'url': url,
            'title': topic,
            'summary': summary,
            'sections': sections,
            'full_text': get_website_text_content(url)
        }
        
    except Exception as e:
        logger.error(f"Error scraping Wikipedia data for {topic}: {str(e)}")
        return {}

def extract_numeric_data(text: str) -> List[float]:
    """
    Extract numeric data from text content
    Useful for analyzing scraped content
    """
    try:
        # Regular expression to find numbers (including decimals)
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        return [float(num) for num in numbers]
    except Exception as e:
        logger.error(f"Error extracting numeric data: {str(e)}")
        return []
