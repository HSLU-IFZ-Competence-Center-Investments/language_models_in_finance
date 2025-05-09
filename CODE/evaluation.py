import pandas as pd
import random
import tinyllamaSLM
import re
import csv
from datetime import datetime
import time


# Define the Person class
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


"""
This function is reading data from the list and storing in object Person.
It returns the list of people. 
"""
def load_data(file_name):
    file_path = "./data/"+ file_name  
    df = pd.read_csv(file_path)
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

def generate_artificial_dataset(people, num_samples=100):
    """
    Generates a synthetic dataset by randomly rearranging first names 
    and last names across entries to create new artificial names.

    Args:
        people (list of Person): The real leadership dataset.
        num_samples (int): Number of synthetic names to generate.

    Returns:
        list of Person: A list containing synthetic Person objects.
    """
    first_names = [person.first_name for person in people]
    last_names = [person.last_name for person in people]

    # Shuffle to create new random combinations
    random.shuffle(first_names)
    random.shuffle(last_names)

    synthetic_people = []
    for i in range(num_samples):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"
        
        synthetic_people.append(Person(
            full_name=full_name,
            first_name=first_name,
            last_name=last_name,
            job_title="Unknown",
            primary_company="Unknown",
            ticker="Unknown"
        ))

    return synthetic_people
    

def is_member(text):
    matches = re.search(r'Answer:\s*(.*)', text)
    if matches:
        answer_only = matches[0]
        if 'Yes' in answer_only:
            return True
        elif "No" in answer_only:
            return False
        else:
            return None

def run_model(model_name, list):
    output = []
    if model_name == "tinyllamaSLM":
        model = tinyllamaSLM

    for person in list:
        full_name = person.full_name
        result = tinyllamaSLM.check_name_in_management_or_board(full_name)
        output.append((person, is_member(result)))

    return output

def synthetic_people_to_dataframe(synthetic_people):
    data = [{
        "Full name": p.full_name,
        "First name": p.first_name,
        "Last name": p.last_name,
        "Job title": p.job_title,
        "Primary company": p.primary_company,
        "Ticker": p.ticker
    } for p in synthetic_people]
    
    df = pd.DataFrame(data)
    return df

#  Real leaders correctly identified
def calculate_true_positives(data, answers):
    result = 0
    for ans in answers:
        if ans[0] in data:
            result += 1
    return result

# Artificial names incorrectly flagged as leaders.
def calculate_false_positives(answers):
    result = 0
    for ans in answers:
        if ans[1] == True:
            result += 1 
    return result

# Artificial names correctly identified as non-leaders.
def calculate_true_negatives(answers):
    result = 0 
    for ans in answers:
        if ans[1] == False:
            result += 1 
    return result

# Real leaders not detected.
def calculate_false_negatives(data, answers):
    result = 0
    for ans in answers:
        if not ans[0] in data:
            result += 1
    return result

def log_metrics(TP, TN, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0  # Avoid division by zero
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0  # Avoid division by zero
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0  # Avoid division by zero
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0  # Avoid division by zero

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [current_time, precision, recall, accuracy, f1_score]
    
    file_name = "results/experiment_results.csv"
    
    # Check if the file exists
    file_exists = False
    try:
        with open(file_name, 'r'):
            file_exists = True
    except FileNotFoundError:
        file_exists = False
    
    # Open the file in append mode and write the row
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header if the file is empty (first time writing)
        if not file_exists:
            writer.writerow(["Date", "Precision", "Recall", "Accuracy", "F1 Score"])
        
        # Append the data row
        writer.writerow(row)
        
    print(f"Metrics logged to {file_name}")


if __name__ == "__main__":
    # real_data = load_data("Namensliste CEOs und BoDs Schweiz.csv")
    # aritfical_data = generate_artificial_dataset(real_data, len(real_data))
    # df = synthetic_people_to_dataframe(aritfical_data)
    # df.to_csv("syntetic_CEOs und BoDs Schweiz.csv", index=False)  # or df.to_excel("synthetic_people.xlsx")
    start = time.perf_counter()

    # Load data files
    real_data = load_data("CEOs und BoDs International.csv")
    artifical_data = load_data("syntetic_CEOs und BoDs International.csv")

    model_name = "tinyllamaSLM"

    answers_real_data = run_model(model_name, real_data)
    answers_aritfical_data = run_model(model_name, artifical_data)


    TP = calculate_true_positives(real_data, answers_real_data)
    FP = calculate_false_positives(answers_aritfical_data)
    TN = calculate_true_negatives(answers_aritfical_data)
    FN = calculate_false_negatives(real_data, answers_real_data)

    log_metrics(TP, FP, TN, FN)

    end = time.perf_counter()

    print(f"Execution time: {end - start:.4f} seconds")


