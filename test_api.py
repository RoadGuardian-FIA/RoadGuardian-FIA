#!/usr/bin/env python3
"""
Simple test script to demonstrate the AI Behavioral Guidelines system
"""

import requests
import json
import time
import subprocess
import os
import sys

API_URL = "http://localhost:8000"


def wait_for_api(timeout=30):
    """Wait for the API to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    return False


def test_endpoint(name, method, endpoint, data=None):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    url = f"{API_URL}{endpoint}"
    
    if method == "GET":
        response = requests.get(url)
    elif method == "POST":
        response = requests.post(url, json=data)
    
    print(f"Status Code: {response.status_code}")
    
    try:
        json_data = response.json()
        print(json.dumps(json_data, indent=2, ensure_ascii=False))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Could not parse JSON response: {e}")
        print(response.text)
    
    return response.status_code == 200


def main():
    """Run all tests"""
    print("AI Behavioral Guidelines - Test Suite")
    print("="*60)
    
    # Check if API is running
    print("\nChecking if API is running...")
    if not wait_for_api(timeout=5):
        print("API is not running. Please start it with:")
        print("  python src/main.py")
        print("or:")
        print("  uvicorn src.main:app --reload")
        return 1
    
    print("✓ API is running")
    
    # Test root endpoint
    test_endpoint("Root Endpoint", "GET", "/")
    
    # Test health endpoint
    test_endpoint("Health Check", "GET", "/health")
    
    # Test protocols endpoint
    test_endpoint("Get All Protocols", "GET", "/protocols")
    
    # Test predict endpoint with various scenarios
    test_cases = [
        {
            "name": "Standard Accident (Highway, Rain, Night)",
            "data": {
                "incident_type": "accident",
                "severity": "high",
                "location_type": "highway",
                "weather": "rain",
                "time_of_day": "night"
            }
        },
        {
            "name": "Traffic Violation (Urban, Clear, Day)",
            "data": {
                "incident_type": "violation",
                "severity": "low",
                "location_type": "urban",
                "weather": "clear",
                "time_of_day": "day"
            }
        },
        {
            "name": "High-Risk Incident (Urban, Snow, Night)",
            "data": {
                "incident_type": "accident",
                "severity": "high",
                "location_type": "urban",
                "weather": "snow",
                "time_of_day": "night"
            }
        },
        {
            "name": "Urban High-Priority (Violation, Clear, Night)",
            "data": {
                "incident_type": "violation",
                "severity": "high",
                "location_type": "urban",
                "weather": "clear",
                "time_of_day": "night"
            }
        }
    ]
    
    for test_case in test_cases:
        test_endpoint(
            f"Predict Protocol: {test_case['name']}", 
            "POST", 
            "/predict", 
            test_case['data']
        )
    
    # Test error handling
    test_endpoint(
        "Error Handling: Invalid Incident Type",
        "POST",
        "/predict",
        {
            "incident_type": "invalid",
            "severity": "high",
            "location_type": "highway",
            "weather": "rain",
            "time_of_day": "night"
        }
    )
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
