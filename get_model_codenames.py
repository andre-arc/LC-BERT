#!/usr/bin/env python3
"""
Script to extract and list all model codenames from save/non_preprocessing and save/processing directories.
"""

from pathlib import Path
import json

def get_model_codenames():
    """Extract all model codenames from both directories."""
    
    base_dir = Path(".")
    non_preprocessing_dir = base_dir / "save" / "non_preprocessing"
    processing_dir = base_dir / "save" / "processing"
    
    codenames = {
        "non_preprocessing": [],
        "processing": []
    }
    
    # Get codenames from non_preprocessing directory
    if non_preprocessing_dir.exists():
        for dataset_dir in non_preprocessing_dir.iterdir():
            if dataset_dir.is_dir():
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir():
                        codenames["non_preprocessing"].append({
                            "dataset": dataset_dir.name,
                            "codename": model_dir.name,
                            "full_path": str(model_dir)
                        })
    
    # Get codenames from processing directory
    if processing_dir.exists():
        for dataset_dir in processing_dir.iterdir():
            if dataset_dir.is_dir():
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir():
                        codenames["processing"].append({
                            "dataset": dataset_dir.name,
                            "codename": model_dir.name,
                            "full_path": str(model_dir)
                        })
    
    return codenames

def print_codenames(codenames):
    """Print codenames in a formatted way."""
    
    print("=" * 80)
    print("MODEL CODENAMES FROM save/non_preprocessing")
    print("=" * 80)
    
    if codenames["non_preprocessing"]:
        for i, model in enumerate(codenames["non_preprocessing"], 1):
            print(f"{i:3d}. Dataset: {model['dataset']}")
            print(f"     Codename: {model['codename']}")
            print(f"     Path: {model['full_path']}")
            print()
    else:
        print("No models found in save/non_preprocessing")
        print()
    
    print("=" * 80)
    print("MODEL CODENAMES FROM save/processing")
    print("=" * 80)
    
    if codenames["processing"]:
        for i, model in enumerate(codenames["processing"], 1):
            print(f"{i:3d}. Dataset: {model['dataset']}")
            print(f"     Codename: {model['codename']}")
            print(f"     Path: {model['full_path']}")
            print()
    else:
        print("No models found in save/processing")
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total models in save/non_preprocessing: {len(codenames['non_preprocessing'])}")
    print(f"Total models in save/processing: {len(codenames['processing'])}")
    print(f"Total models overall: {len(codenames['non_preprocessing']) + len(codenames['processing'])}")
    print("=" * 80)

def save_codenames_to_file(codenames, output_file="model_codenames.json"):
    """Save codenames to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(codenames, f, indent=2)
    print(f"\nCodenames saved to {output_file}")

def save_codenames_to_txt(codenames, output_file="model_codenames.txt"):
    """Save codenames to a text file."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL CODENAMES FROM save/non_preprocessing\n")
        f.write("=" * 80 + "\n\n")
        
        if codenames["non_preprocessing"]:
            for i, model in enumerate(codenames["non_preprocessing"], 1):
                f.write(f"{i:3d}. Dataset: {model['dataset']}\n")
                f.write(f"     Codename: {model['codename']}\n")
                f.write(f"     Path: {model['full_path']}\n\n")
        else:
            f.write("No models found in save/non_preprocessing\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("MODEL CODENAMES FROM save/processing\n")
        f.write("=" * 80 + "\n\n")
        
        if codenames["processing"]:
            for i, model in enumerate(codenames["processing"], 1):
                f.write(f"{i:3d}. Dataset: {model['dataset']}\n")
                f.write(f"     Codename: {model['codename']}\n")
                f.write(f"     Path: {model['full_path']}\n\n")
        else:
            f.write("No models found in save/processing\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total models in save/non_preprocessing: {len(codenames['non_preprocessing'])}\n")
        f.write(f"Total models in save/processing: {len(codenames['processing'])}\n")
        f.write(f"Total models overall: {len(codenames['non_preprocessing']) + len(codenames['processing'])}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Codenames saved to {output_file}")

if __name__ == "__main__":
    codenames = get_model_codenames()
    print_codenames(codenames)
    save_codenames_to_file(codenames)
    save_codenames_to_txt(codenames)
