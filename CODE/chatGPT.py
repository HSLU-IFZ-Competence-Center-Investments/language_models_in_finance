import subprocess
import sys
import time
import urllib
from utils.chatutils import AssistantSession
from utils.datamanager import get_companyfilepaths,cfg
from datetime import datetime
import os
import pandas as pd
import unicodedata
from tqdm import tqdm
import csv
from openai import OpenAI

# Create a client (you can put this outside the function for reuse)
client = OpenAI(api_key=api_key) # write your OpenAI key here

max_chars_total = 3 * 1000000
gpt_name = "Compliance expert"
instruction = "Determine whether the given person is part of the management team or board of a publicly listed company, based on provided context. Returns: 'Yes' or 'No'"
model="gpt-4.1"
temperature=0

class Person:
    def __init__(self, full_name, first_name, last_name, job_title, primary_company, ticker):
        self.full_name = full_name
        self.first_name = first_name
        self.last_name = last_name
        self.job_title = job_title
        self.primary_company = primary_company
        self.ticker = ticker

    def __str__(self):
        return f"{self.full_name}, {self.job_title} at {self.primary_company} ({self.ticker})"



def check_name_in_management_or_board(name, rag, filtered_output=None):
    if rag:
        if filtered_output is None:
            raise ValueError("Filtered output must be provided when RAG is enabled.")

        # Look up the context for the given name
        person_entry = next((item for item in filtered_output if item['full_name'] == name), None)
        if person_entry is None:
            print(f"⚠️ No context found for {name}, using empty context.")
            context = ""
        else:
            context = "\n".join(person_entry['filtered_documents'])

    else:
        context = ""

    #print(f"🧠 Context length: {len(context)} characters")
    #print(name)
    #print(f"⚠️ Context: {context}")
    result = generate_answer(context, name)
    return result

def load_data(file_name):
    encoded_file_name = urllib.parse.quote(file_name)
    file_path = folder_path + encoded_file_name

    df = pd.read_csv(file_path, sep=",", quotechar='"', escapechar="\\", engine="python", on_bad_lines='skip')

    people = []
    for index, row in df.iterrows():
        person = Person(
            full_name=row['Full Name'],
            first_name=row['First Name'],
            last_name=row['Last Name'],
            job_title=row['Job Title'],
            primary_company=row['Primary Company'],
            ticker=row['Ticker']
        )
        people.append(person)

    return people

def calculate_true_positives(data, answers):
    known_names = {p.full_name for p in data}
    return sum(1 for p, is_member in answers if p.full_name in known_names and is_member)

def calculate_false_positives(answers):
    return sum(1 for p, is_member in answers if is_member)

def calculate_true_negatives(answers):
    return sum(1 for _, is_member in answers if not is_member)

def calculate_false_negatives(data, answers):
    known_names = {p.full_name for p in data}
    return sum(1 for p, is_member in answers if p.full_name in known_names and not is_member)

def is_member(text):
    if "yes" in text.lower():
      return True
    elif "no" in text.lower():
      return False
    else:
      None 

def run_model(model_name, people_list, rag, filtered_output=None):
    output = []
    progress = tqdm(people_list, desc=f"🔍 Running model: {model_name}", unit="person")

    for person in progress:
        full_name = person.full_name
        result = check_name_in_management_or_board(full_name, rag, filtered_output)
        member_status = is_member(result)
        # if member_status is True:
        #     status_str = f"[{model_name}] {full_name}: ✅ Member"
        # elif member_status is False:
        #     status_str = f"[{model_name}] {full_name}: ❌ Not a Member"
        # else:
        #     status_str = f"[{model_name}] {full_name}: ❓ Unclear result: '{result}'"
        
        # tqdm.write(status_str)
        # sys.stdout.flush()
        output.append((person, member_status))

    return output

def check_for_unclear_results(data):
    cnt = 0
    for _, is_member in data:
        if is_member == None:
            cnt += 1

    return cnt

def process_data(dataset_type, rag, model_name, filtered_output_real, filtered_output_synthetic):
    start = time.perf_counter()
    if dataset_type == "international":
        real_data = load_data("CEOs und BoDs International.csv")
        artifical_data = load_data("syntetic_CEOs und BoDs International.csv")
    elif dataset_type == "switzerland":
        real_data = load_data("Namensliste CEOs und BoDs Schweiz.csv")
        artifical_data = load_data("syntetic_CEOs und BoDs Schweiz.csv")
    elif dataset_type == "switzerland_nw":  # Nebenwerte
        real_data = load_data("Namensliste CEOs und BoDs Nebenwerte Schweiz.csv")
        artifical_data = load_data("syntetic_CEOs und BoDs Nebenwerte Schweiz.csv")
    else:
        raise ValueError("Invalid dataset_type. Choose from: 'international', 'switzerland', 'switzerland_nw'.")

    answers_real_data = run_model(model_name, real_data, rag, filtered_output_real)
    answers_artifical_data = run_model(model_name, artifical_data, rag, filtered_output_synthetic)

    TP = calculate_true_positives(real_data, answers_real_data)
    FP = calculate_false_positives(answers_artifical_data)
    TN = calculate_true_negatives(answers_artifical_data)
    FN = calculate_false_negatives(real_data, answers_real_data)

    unclear_results_rd = check_for_unclear_results(answers_real_data)
    unclear_results_ad = check_for_unclear_results(answers_artifical_data)

    end = time.perf_counter()
    execution_time = end - start

    print(f"Execution time: {execution_time:.2f} seconds")
    log_metrics(TP, FP, TN, FN, model_name, dataset_type, rag, execution_time, unclear_results_rd, unclear_results_ad)


def get_gpu_name():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            encoding='utf-8'
        )
        return output.strip()
    except Exception as e:
        return f"Kein GPU gefunden: {e}"

def log_metrics(TP, FP, TN, FN, model_name, dataset_type, rag, execution_time, unclear_results_rd, unclear_results_ad):
    gpu_name = get_gpu_name()
    if dataset_type == "international":
        list_type = "International"
    elif dataset_type == "switzerland":
        list_type = "Switzerland"
    elif dataset_type == "switzerland_nw":
        list_type = "Switzerland Nebenwerte"
    else:
        list_type = "Unknown"

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [current_time, model_name, list_type, rag, TP, FP, TN, FN, unclear_results_rd, unclear_results_ad, f"{precision:.4f}", f"{recall:.4f}", f"{accuracy:.4f}", f"{f1_score:.4f}", f"{execution_time:.2f}", gpu_name]
    # print()
    # print("Date, Model name, Data scope, RAG, TP, FP, TN, FN, Precision, Recall, Accuracy, F1 Score, Execution time, GPU Name")
    # print(row)

    # Open the file in append mode and write the row

    file_name = os.path.join("./experiment_results-chatgpt.csv")
    file_exists = os.path.exists(file_name)


    with open(file_name, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header if the file is empty (first time writing)
        if not file_exists:
            writer.writerow(["Date", "Model name", "Data scope", "RAG", "TP", "FP", "TN", "FN", "Unclear result in real dataset", "Unclear result in artificial dataset", "Precision", "Recall", "Accuracy", "F1 Score", "Execution time", "GPU name"])

        # Append the data row
        writer.writerow(row)

    print(f"Metrics logged to {file_name}")


def generate_answer(context, name):

    prompt = f"""
        Context:
        {context}
        
        Question:
        Is {name} a member of the management team (e.g. CEO, CFO, CTO, COO, CAO, CIO, EVP, head, director) or board of directors of a publicly listed company?
        
        Answer with exactly one word: Yes or No.
        Answer:"""
 
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a classification assistant. Always answer exactly Yes or No."},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=temperature,
        max_tokens=20,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    answer = response.choices[0].message.content.strip()
    return answer


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


def filter_relevant_paragraphs(doc, name, keywords=None, max_chars_per_doc=None):
    if keywords is None:
        name_parts = name.lower().split()
        keywords = [
            # German roots
            "verwaltungsra", "präsident", "geschäftsleitung", "konzernleitung",
            "unternehmer", "direktor", "firmenleitung", "leiter",
            "geschäftsführer", "stiftungsrat", "exekutiv", "verantwort", "aufsichtsra", "chef"

            # English roots
            "director", "board", "chief", "executive", "ceo", "cto", "cfo", "coo", "cao", "evp", "cio", "head",
            "committee", "leader", "supervisor", "president", "chair",
            "management", "managing", "manager", "founder",
            "entrepreneur", "c-level", "c-suite"
        ]
    else:
        name_parts = name.lower().split()

    # Normalize keywords and name parts
    keywords = [normalize_text(k.lower()) for k in keywords]
    name_parts = [normalize_text(k.lower()) for k in name_parts]

    if doc != 'empty':
        paragraphs = [p.strip() for p in doc.split("\n") if len(p.strip()) > 20]
        if paragraphs:
            scored_paragraphs = []
            for p in paragraphs:
                normalized_p = normalize_text(p.lower())
                # Check if any name part AND at least one keyword are in the paragraph
                name_found = any(name_part in normalized_p for name_part in name_parts)
                keyword_found = any(keyword in normalized_p for keyword in keywords)
                if name_found and keyword_found:
                    # Simple scoring: count all keyword hits
                    score = sum(normalized_p.count(k) for k in keywords)
                    scored_paragraphs.append((score, p))

            sorted_paragraphs = sorted(scored_paragraphs, key=lambda x: x[0], reverse=True)
            selected = []
            total_chars = 0
            for _, p in sorted_paragraphs:
                if len(p) > max_chars_per_doc:
                    selected.append(p[:max_chars_per_doc])
                    break
                if total_chars + len(p) > max_chars_per_doc:
                    break
                selected.append(p)
                total_chars += len(p) + 1

            return "\n".join(selected) if selected else False
        return False
    return False

def load_and_filter_documents(file_path):
    df = pd.read_csv(file_path)
    filtered_data = []

    for index, row in df.iterrows():
        name = row['full_name']
        try:
            documents = eval(row['raw_documents'])  # Parse stringified list
        except Exception as e:
            print(f"❌ Error parsing documents for {name}: {e}")
            documents = []

        num_docs = len(documents)
        if num_docs == 0:
            max_chars_per_doc = 0
        else:
            max_chars_per_doc = max_chars_total // num_docs

        #print(max_chars_per_doc)
        filtered_docs = []
        for doc in documents:
            filtered = filter_relevant_paragraphs(doc, name, max_chars_per_doc=max_chars_per_doc)
            if filtered:
                filtered_docs.append(filtered)

        filtered_data.append({
            "full_name": name,
            "filtered_documents": filtered_docs
        })

    return filtered_data


def get_filtered_outputs(dataset_type):
    base_url = (
        "https://raw.githubusercontent.com/HSLU-IFZ-Competence-Center-Investments/compliance_data_set/main/Raw%20Data%20Strict/"
    )

    if dataset_type == "international":
        real_file = "CEOs%20und%20BoDs%20International_offline_raw_documents_international_real_strict.csv"
        synthetic_file = "syntetic_CEOs%20und%20BoDs%20International_offline_raw_documents_international_synthetic_strict.csv"
    elif dataset_type == "switzerland":
        real_file = "Namensliste%20CEOs%20und%20BoDs%20Schweiz_offline_raw_documents_real_strict.csv"
        synthetic_file = "syntetic_CEOs%20und%20BoDs%20Schweiz_offline_raw_documents_synthetic_strict.csv"
    elif dataset_type == "switzerland_nw":
        real_file = "Namensliste%20CEOs%20und%20BoDs%20Nebenwerte%20Schweiz_offline_raw_documents_Nebenwerte_real_strict.csv"
        synthetic_file = "syntetic_CEOs%20und%20BoDs%20Nebenwerte%20Schweiz_offline_raw_documents_Nebenwerte_synthetic_strict.csv"
    else:
        raise ValueError("Invalid dataset_type. Choose from: 'international', 'switzerland', 'switzerland_nw'.")

    file_path_real = base_url + real_file
    file_path_synthetic = base_url + synthetic_file

    filtered_output_real = load_and_filter_documents(file_path_real)
    filtered_output_synthetic = load_and_filter_documents(file_path_synthetic)

    return filtered_output_real, filtered_output_synthetic

folder_path = 'https://raw.githubusercontent.com/HSLU-IFZ-Competence-Center-Investments/compliance_data_set/refs/heads/main/'
dataset_types = ["international", "switzerland", "switzerland_nw"]
rag_options = [False, True]

if __name__ == "__main__":
    for dataset_type in dataset_types:
        for rag in rag_options:
            print("\n---")
            print(f"Running: dataset_type = {dataset_type}, RAG = {rag}")

            # Get the filtered documents for the current dataset type
            filtered_output_real, filtered_output_synthetic = get_filtered_outputs(dataset_type)

            # Call your processing function
            process_data(dataset_type, rag, model, filtered_output_real, filtered_output_synthetic)