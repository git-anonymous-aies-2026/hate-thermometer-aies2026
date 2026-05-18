import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import time
from collections import defaultdict
# ===========================================================================================
# EXPERIMENT 1: Dynamic Slur Table Extraction and JSON Export
# ============================================================================================

def export_rsdb_to_json_beautifully():
    url = "http://www.rsdb.org/full"

    print("Downloading page...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to download page.")
        return
    
    print("Parsing HTML...")
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")

    if not table:
        print("No table found.")
        return

    rows = table.find_all("tr")
    print(f"Found {len(rows)} rows.")

    slur_dict = []
    # slur_dict = defaultdict(list)
    for i, row in enumerate(rows[1:], start=1):
        cols = row.find_all("td")

        if len(cols) < 3:
            continue 

        slur_dict.append({
        'slur ': cols[0].get_text(strip=True).lower(),
        'target' : cols[1].get_text(strip=True),
        'explanation' : cols[2].get_text(strip=True)})

        if i % 100 == 0:
            print(f"Processed {i} rows...")

    with open("rsdb_slurs_with_dups.json", "w", encoding="utf-8") as f:
        json.dump(slur_dict, f, indent=4, ensure_ascii=False)

    print(f"Finished. Exported {len(slur_dict)} unique slurs.")


import json
with open('rsdb_slurs_with_dups.json', 'r') as file:
    slur_database = json.load(file)
    print(slur_database[0:2])

# Sorting by length (longest first) to prevent partial matching errors e.g matching 'african' in "african't"
slur_database_sorted = sorted(slur_database, key=lambda x: len(x['slur']), reverse=True)

# Function to apply dynamic slur tags to a given text using the slur database extracted from RSDB
def apply_dynamic_slur_tags(text, slur_db):
    tagged_text = text
    for entry in slur_db:
        slur = entry['slur '].strip()
        target = entry['target'].strip()
        explanation = entry['explanation'].strip()
        
        # creating the tag containing the side knowledge
        replacement_tag = f"<slur target='{target}' explanation='{explanation}'>{slur}</slur>"
        
        # Use regex with \b to delimneated the slur as a whole word, and re.IGNORECASE for case-insensitive matching
        pattern = re.compile(rf"\b{re.escape(slur)}\b", re.IGNORECASE)
        tagged_text = pattern.sub(replacement_tag, tagged_text)
        
    return tagged_text

def main():
    # export_rsdb_to_json()
    export_rsdb_to_json_beautifully()

if __name__ == "__main__":
    main()