#!/usr/bin/env python3
"""Test script to verify API response and frontend data handling."""

import json
import requests

def test_api_response():
    """Test the API endpoint directly."""
    url = "http://localhost:5000/api/train"
    data = {
        "omega": 1.0,
        "t_max": 10.0,
        "epochs": 100,
        "hidden_dim": 16,
        "num_domain_points": 100,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "lambda_ic": 1.0,
        "lambda_physics": 1.0
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print("=" * 60)
    print("API Response Analysis")
    print("=" * 60)
    print(f"Keys in response: {list(result.keys())}")
    print(f"t length: {len(result['t'])}")
    print(f"x_pred length: {len(result['x_pred'])}")
    print(f"x_analytical length: {len(result['x_analytical'])}")
    print(f"velocity length: {len(result['velocity'])}")
    print(f"residual length: {len(result['residual'])}")
    print()
    print("Sample t values (first 10):")
    print(result['t'][:10])
    print()
    print("Sample x_pred values (first 10):")
    print(result['x_pred'][:10])
    print()
    print("Sample x_analytical values (first 10):")
    print(result['x_analytical'][:10])
    print()
    print("Sample velocity values (first 10):")
    print(result['velocity'][:10])
    print()
    print("Final loss:", result['final_loss'])
    print("=" * 60)
    
    return result

def test_frontend_data_mapping():
    """Test the data mapping that happens in the frontend."""
    result = test_api_response()
    
    print("\n" + "=" * 60)
    print("Frontend Data Mapping Simulation")
    print("=" * 60)
    
    # This simulates what the frontend does (Python equivalent of JavaScript map)
    prediction_data = [{'x': result['t'][i], 'y': result['x_pred'][i]} for i in range(len(result['x_pred']))]
    
    # Show first 5 data points
    print("\nFirst 5 prediction data points:")
    for i in range(5):
        point = prediction_data[i]
        print(f"  x={point['x']:.4f}, y={point['y']:.4f}")
    
    # Check if x values are all the same (the bug)
    unique_x_values = len(set(result['t']))
    print(f"\nNumber of unique t values: {unique_x_values}")
    
    # Check if t values are actually different
    if len(set(result['t'])) == 1:
        print("ERROR: All t values are the same! This is the bug.")
    else:
        print("OK: t values are different.")
    
    print("=" * 60)

if __name__ == "__main__":
    test_frontend_data_mapping()