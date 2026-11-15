import json

def extract_matching_records(source_file, lookup_file, output_file):
    """
    Extract destination IPs from source file, then find matching records in lookup file
    and write them to output file.
    
    Args:
        source_file (str): Path to file containing records to extract dst IPs from (Osaka.txt.txt)
        lookup_file (str): Path to file to search for matching dst IPs (Osaka_completed_final.txt)
        output_file (str): Path to output file (Osaka_completed_v4_anony_final.txt)
    """
    
    # Step 1: Extract all dst IPs from source file
    dst_ips = set()
    print(f"Reading destination IPs from {source_file}...")
    
    with open(source_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                dst_ip = record.get('dst')
                if dst_ip:
                    dst_ips.add(dst_ip)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num} in source file: {e}")
                continue
    
    print(f"Found {len(dst_ips)} unique destination IPs in source file")
    
    # Step 2: Search for matching records in lookup file
    matched_count = 0
    print(f"Searching for matching records in {lookup_file}...")
    
    with open(lookup_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                dst_ip = record.get('dst')
                
                if dst_ip and dst_ip in dst_ips:
                    outfile.write(json.dumps(record) + '\n')
                    matched_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num} in lookup file: {e}")
                continue
    
    print(f"Found {matched_count} matching records")
    print(f"Written to {output_file}")
    
    return matched_count

# Alternative version with progress indicator for large files
def extract_matching_records_with_progress(source_file, lookup_file, output_file):
    """
    Same as above but with progress indicator for large files.
    """
    # Step 1: Extract all dst IPs from source file
    dst_ips = set()
    print(f"Reading destination IPs from {source_file}...")
    
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            record = json.loads(line)
            dst_ip = record.get('dst')
            if dst_ip:
                dst_ips.add(dst_ip)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse line {i} in source file: {e}")
            continue
        
        # Progress indicator for large files
        if i % 10000 == 0:
            print(f"Processed {i} lines from source file...")
    
    print(f"Found {len(dst_ips)} unique destination IPs in source file")
    
    # Step 2: Search for matching records in lookup file
    matched_count = 0
    print(f"Searching for matching records in {lookup_file}...")
    
    with open(lookup_file, 'r', encoding='utf-8') as infile:
        lookup_lines = infile.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(lookup_lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                dst_ip = record.get('dst')
                
                if dst_ip and dst_ip in dst_ips:
                    outfile.write(json.dumps(record) + '\n')
                    matched_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {i} in lookup file: {e}")
                continue
            
            # Progress indicator for large files
            if i % 10000 == 0:
                print(f"Processed {i} lines from lookup file, found {matched_count} matches...")
    
    print(f"Found {matched_count} matching records")
    print(f"Written to {output_file}")
    
    return matched_count

# If you want to also see which IPs were matched and which weren't
def extract_matching_records_with_debug(source_file, lookup_file, output_file):
    """
    Same as above but with debug information about matched/unmatched IPs.
    """
    # Step 1: Extract all dst IPs from source file
    dst_ips = set()
    print(f"Reading destination IPs from {source_file}...")
    
    with open(source_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                dst_ip = record.get('dst')
                if dst_ip:
                    dst_ips.add(dst_ip)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num} in source file: {e}")
                continue
    
    print(f"Found {len(dst_ips)} unique destination IPs in source file")
    print(f"Sample IPs: {list(dst_ips)[:5]}")
    
    # Step 2: Search for matching records in lookup file
    matched_count = 0
    matched_ips = set()
    
    print(f"Searching for matching records in {lookup_file}...")
    
    with open(lookup_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                dst_ip = record.get('dst')
                
                if dst_ip and dst_ip in dst_ips:
                    outfile.write(json.dumps(record) + '\n')
                    matched_count += 1
                    matched_ips.add(dst_ip)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num} in lookup file: {e}")
                continue
    
    # Show statistics
    unmatched_ips = dst_ips - matched_ips
    
    print(f"\n--- Statistics ---")
    print(f"Total unique IPs in source: {len(dst_ips)}")
    print(f"Matched records found: {matched_count}")
    print(f"Unique IPs matched: {len(matched_ips)}")
    print(f"Unique IPs NOT matched: {len(unmatched_ips)}")
    
    if unmatched_ips:
        print(f"\nFirst 10 unmatched IPs: {list(unmatched_ips)[:10]}")
    
    print(f"\nWritten to {output_file}")
    
    return matched_count, matched_ips, unmatched_ips

# Usage
if __name__ == "__main__":
    source_file = "Osaka.txt.txt"
    lookup_file = "Osaka_completed_final.txt"
    output_file = "Osaka_completed_final_final.txt"
    
    # Choose one of the functions:
    
    # Basic version
    extract_matching_records(source_file, lookup_file, output_file)
    
    # Or with progress indicator (good for large files)
    # extract_matching_records_with_progress(source_file, lookup_file, output_file)
    
    # Or with debug information
    # extract_matching_records_with_debug(source_file, lookup_file, output_file)
