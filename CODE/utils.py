
import requests
from bs4 import BeautifulSoup
import evaluation
from urllib.parse import quote

GOOGLE_API_KEY = "add yours"
CSE_ID = "add yours"

def google_search(query, api_key, cse_id): # STRICT SEARCH
    encoded_query = quote(query)
    url = (
        f"https://www.googleapis.com/customsearch/v1?"
        f"q={encoded_query}&key={api_key}&cx={cse_id}&dateRestrict=y[1]"
    )
    response = requests.get(url)
    return response.json()




def search_name_on_internet(name): # STRICT SEARCH, none strict search -> query = f"{name}"
    query = f'"{name}"' # f"{name}" non strict and f'"{name}"' strict # Potentially adjust to f"{name} management team OR board of directors"
    results = google_search(query, GOOGLE_API_KEY, CSE_ID)
    links = [item['link'] for item in results.get('items', [])]
    return links

def scrape_page(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException:
        return ""
    

def filter_relevant_paragraphs(doc, name, keywords=None, max_chars_per_doc=2000):
    if keywords is None:
        keywords = [name.lower()]

    paragraphs = [p.strip() for p in doc.split("\n") if len(p.strip()) > 50]
    
    # Score paragraphs by keyword hits
    scored_paragraphs = []
    for p in paragraphs:
        score = sum(p.lower().count(k.lower()) for k in keywords)
        if score > 0:
            scored_paragraphs.append((score, p))
    
    # Sort paragraphs by score descending
    sorted_paragraphs = sorted(scored_paragraphs, key=lambda x: x[0], reverse=True)

    # Add paragraphs until max_chars_per_doc is reached
    selected = []
    total_chars = 0
    for _, p in sorted_paragraphs:
        if total_chars + len(p) > max_chars_per_doc:
            break
        selected.append(p)
        total_chars += len(p) + 1  # +1 for newline

    return "\n".join(selected)

def retrieve_documents(links, name):
    documents = []
    for link in links:
        content = scrape_page(link)
        if content:
            filtered = filter_relevant_paragraphs(content, name)
            if filtered:
                documents.append(filtered)
    return documents


