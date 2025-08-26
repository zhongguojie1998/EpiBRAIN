#!/usr/bin/env python3

import os
import sys

def check_pt_files():
    sequences_file = "/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Data/basel_ganglia_complete_v1/sequences.bed"
    data_dir = "/gpfs/commons/groups/ren_lab/guojiezhong/BICAN/Data/basel_ganglia_complete_v1/data"
    
    missing_files = []
    empty_files = []
    total_lines = 0
    
    with open(sequences_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 3:
                continue
                
            chr_name = parts[0]
            start = parts[1]
            end = parts[2]
            
            # Construct expected .pt filename
            pt_filename = f"{chr_name}_{start}_{end}.pt"
            pt_filepath = os.path.join(data_dir, pt_filename)
            
            total_lines += 1
            
            # Check if file exists
            if not os.path.exists(pt_filepath):
                missing_files.append((line_num, line, pt_filename))
                print(f"MISSING: Line {line_num}: {line} -> {pt_filename}")
            else:
                # Check if file is empty
                try:
                    file_size = os.path.getsize(pt_filepath)
                    if file_size == 0:
                        empty_files.append((line_num, line, pt_filename))
                        print(f"EMPTY: Line {line_num}: {line} -> {pt_filename}")
                except OSError as e:
                    print(f"ERROR: Line {line_num}: {line} -> {pt_filename} - {e}")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total sequences checked: {total_lines}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Empty files: {len(empty_files)}")
    print(f"Valid files: {total_lines - len(missing_files) - len(empty_files)}")
    
    if missing_files or empty_files:
        print(f"\nPROBLEMS FOUND:")
        if missing_files:
            print(f"  {len(missing_files)} missing files")
        if empty_files:
            print(f"  {len(empty_files)} empty files")
    else:
        print("All files exist and have content!")

if __name__ == "__main__":
    check_pt_files()