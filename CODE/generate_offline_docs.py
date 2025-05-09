import utils
import evaluation
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
import unicodedata

max_chars_list = {
    'TinyLlama-1.1B-Chat-v1.0': 600,
    'phi-2': 1500,
    'zephyr-7b-alpha': 1200,
    'Mistral-7B-Instruct-v0.1': 3000
}

max_chars_per_doc = max(max_chars_list.values())

def normalize_text(text):
    replacements = {
        "ä": "ae", "ö": "oe", "ü": "ue",
        "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
        "ß": "ss",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def filter_relevant_paragraphs(doc, name, keywords=None):
    # if keywords is None:
    #     keywords = [name.lower()]
    
    if keywords is None:
        keywords = name.lower().split()
    # Extend with board/management-specific keywords
        keywords += [
            # German roots
            "verwaltungsrat", "präsident", "geschäftsleitung", "konzernleitung",
            "unternehmer", "direktor", "firmenleitung", "leiter", "manager",
            "geschäftsführer", "stiftungsrat", "exekutivrat", "verantwort", "aufsichtsrat",
            
            # English roots
            "board of directors", "board member", "chair", "chief executive", "ceo", "cto", "cfo",
            "executive board", "executive committee", "leadership team", "leader",
            "management team", "managing director", "director", "founder",
            "entrepreneur", "head of", "c-level", "c-suite"
        ]
    
    # Normalize keywords
    keywords = [normalize_text(k.lower()) for k in keywords]

    if doc != 'empty':  # this means that soap returned empty text
        paragraphs = [p.strip() for p in doc.split("\n") if len(p.strip()) > 20] 
        print(paragraphs) 
        if paragraphs != []:
            # Score paragraphs by keyword hits
            scored_paragraphs = []
            for p in paragraphs:
                normalized_p = normalize_text(p.lower())
                score = sum(normalized_p.count(k.lower()) for k in keywords)
                if score > 0:
                    scored_paragraphs.append((score, p))

            # Sort paragraphs by score descending
            sorted_paragraphs = sorted(scored_paragraphs, key=lambda x: x[0], reverse=True)
            # Add paragraphs until max_chars_per_doc is reached
            selected = []
            total_chars = 0
            for _, p in sorted_paragraphs:
                if len(p) > max_chars_per_doc:
                    # If a single paragraph is too big, truncate it
                    selected.append(p[:max_chars_per_doc])
                    break
                if total_chars + len(p) > max_chars_per_doc:
                    break
                selected.append(p)
                total_chars += len(p) + 1  # +1 for newline

            temp = "\n".join(selected)
            return temp
        return False
    else:
        return False
    
    
def scrape_page(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            print(f"❌ Skipping non-HTML content at {url} (Content-Type: {content_type})")
            return "error"

        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator="\n").strip()
        if not text:
            return "empty"
        return text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return "error"

    except Exception as e:
        print(f"⚠️ Parsing failed for {url}: {e}")
        return "error"    


def retrieve_documents(links, name):
    documents = []
    for link in links:
        content = scrape_page(link)
        if content != 'error':
            filtered = filter_relevant_paragraphs(content, name)
            if filtered:
                documents.append(filtered)
    return documents


def generate_offline_file(data, file_name_prefix):
    rows = []
    for person in data:
        name = person.full_name
        links = utils.search_name_on_internet(name)
        if links != []:
            documents = retrieve_documents(links, name)
        else:
            raise Exception("Quota exceeded for quota metric 'Queries' and limit 'Queries per day'")

        rows.append([name, documents])

    df = pd.DataFrame(rows, columns=['full_name', 'documents'])
    file_path = './offline_data/'+file_name_prefix+'_offline_documents_v2.csv' # Offline file with v2 of filter_relevant_paragraphs
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if the file already exists
    if os.path.isfile(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

    print(f"✅ Results saved to {file_path}")
   
    
if __name__ == "__main__":
    real_data_name = "Namensliste CEOs und BoDs Schweiz.csv"
    real_data = evaluation.load_data(real_data_name)

    name_without_csv = real_data_name.replace(".csv", "")
    safe_name = name_without_csv.replace("/", "_")  # replace slash to avoid filename issues
    #generate_offline_file(real_data[63:], safe_name)
    generate_offline_file(real_data, safe_name)

    # name = "Hernan Rodriguez Wilson"
    # links = utils.search_name_on_internet(name)
    # documents = retrieve_documents(links, name)
    # print("Document")
    # print(documents)