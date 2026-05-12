# Import the DuckDuckGo search tool from LangChain community package
from ddgs import DDGS

# Import the generic Tool wrapper from LangChain
from langchain.tools import tool

# Standard libraries for scraping and saving
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

# Save-to-text tool: saves the output to a text file
@tool
def save_to_txt(data: str, filename: str = "leads_output.txt"):
    """Save-to-text tool: saves the output to a text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Leads Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    # Open the file in append mode so it keeps growing over time
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

# Scrape raw text from a website
@tool
def scrape_website(url: str) -> str:
    """Scrape raw text from a website"""
    try:
        # Send GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad HTTP codes

        # Parse and clean up the raw HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        # Limit to 5000 characters for performance and API limits
        return text[:5000]
    except Exception as e:
        return f"Error scraping website: {e}"

# Generate search queries to look for IT services related to a company
def generate_search_queries(company_name: str) -> list[str]:
    """Generate search queries to look for IT services related to a company"""
    keywords = ["IT Services", "managed IT", "technology solutions"]
    return [f"{company_name} {keyword}" for keyword in keywords]

# Combined search and scrape operation for a company
@tool
def search_and_scrape(company_name: str) -> str:
    """
    Search company information online and scrape websites.
    """

    try:

        query = f"{company_name} official website"

        urls = []

        # Direct DDGS search
        with DDGS() as ddgs:

            results = ddgs.text(
                query,
                max_results=5
            )

            for r in results:

                if "href" in r:
                    urls.append(r["href"])

        if not urls:
            return "No search results found."

        scraped_results = []

        for url in urls[:2]:

            scraped_text = scrape_website.invoke({
                "url": url
            })

            scraped_results.append(
                f"\nURL: {url}\n{scraped_text}"
            )

        return "\n".join(scraped_results)

    except Exception as e:

        return f"Search error: {str(e)}"