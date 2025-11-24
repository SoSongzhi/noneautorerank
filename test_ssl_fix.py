#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify SSL fix works
"""
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import requests
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Test the Prosit API with SSL verification disabled
url = "https://koina.wilhelmlab.org/v2/models/Prosit_2025_intensity_MultiFrag/infer"
payload = {
    "id": "test",
    "inputs": [
        {
            "name": "peptide_sequences",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": ["PEPTIDE"]
        },
        {
            "name": "precursor_charges",
            "shape": [1, 1],
            "datatype": "INT32",
            "data": [2]
        },
        {
            "name": "fragmentation_types",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": ["HCD"]
        }
    ]
}

print("Testing Prosit API connection with SSL verification disabled...")
try:
    response = requests.post(
        url,
        json=payload,
        headers={'Content-Type': 'application/json'},
        timeout=30,
        verify=False  # Disable SSL verification
    )
    print(f"[OK] Success! Status code: {response.status_code}")
    if response.status_code == 200:
        print("[OK] Prosit API is working correctly!")
    else:
        print(f"[WARN] Got status code {response.status_code}")
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"[ERROR] Error: {e}")