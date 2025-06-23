#!/usr/bin/env python3
"""
Clean up the safety_check_records.csv file
Remove duplicate headers and fix format issues
"""

import os
import re

def clean_csv():
    """Clean the safety_check_records.csv file"""
    csv_file = "safety_check_records.csv"
    
    if not os.path.exists(csv_file):
        print("❌ CSV file not found")
        return
    
    print("🧹 Cleaning CSV file...")
    
    # Read the file
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    
    # Clean up the lines
    cleaned_lines = []
    current_date = None
    header_added = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check if it's a date header
        if line.startswith("----") and line.endswith("----"):
            date_match = re.search(r"---- (\d{4}-\d{2}-\d{2}) ----", line)
            if date_match:
                date = date_match.group(1)
                if date != current_date:
                    current_date = date
                    header_added = False
                    cleaned_lines.append(f"---- {date} ----\n")
                    cleaned_lines.append("Batch Size,Defect Rate,Timestamp\n")
                continue
        
        # Check if it's the column header
        if line == "Batch Size,Defect Rate,Timestamp":
            if not header_added:
                header_added = True
            continue
        
        # Check if it's a valid data entry (3 comma-separated values)
        parts = line.split(',')
        if len(parts) == 3:
            try:
                batch_size = float(parts[0])
                defect_rate = float(parts[1])
                timestamp = parts[2]
                
                # Validate timestamp format
                if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', timestamp):
                    cleaned_lines.append(f"{batch_size},{defect_rate:.2f},{timestamp}\n")
                else:
                    print(f"⚠️ Skipping invalid timestamp: {timestamp}")
            except ValueError:
                print(f"⚠️ Skipping invalid data line: {line}")
        else:
            print(f"⚠️ Skipping malformed line: {line}")
    
    # Write the cleaned file
    with open(csv_file, 'w') as f:
        f.writelines(cleaned_lines)
    
    print(f"✅ CSV file cleaned! {len(cleaned_lines)} lines written")
    
    # Show the cleaned content
    print("\n📋 Cleaned CSV content:")
    with open(csv_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    clean_csv() 