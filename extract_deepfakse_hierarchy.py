import os
import json

def get_directory_hierarchy(root_path, target_folders):
    hierarchy = {}
    
    for folder_name in target_folders:
        folder_path = os.path.join(root_path, folder_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            hierarchy[folder_name] = _build_tree(folder_path)
        else:
            print(f"Warning: Folder not found or not a directory -> {folder_path}")
            
    return hierarchy

def _build_tree(path):
    tree = {"type": "directory", "name": os.path.basename(path), "children": []}
    
    try:
        entries = os.listdir(path)
    except PermissionError:
        return tree
        
    for entry in entries:
        # Ignore .zep (and .zip just in case it was a typo)
        if entry.endswith('.zep') or entry.endswith('.zip'):
            continue
            
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            tree["children"].append(_build_tree(full_path))
        else:
            tree["children"].append({"type": "file", "name": entry})
            
    return tree

def count_mp4_in_tree(tree):
    count = 0
    if tree["type"] == "file":
        if tree["name"].endswith(".mp4"):
            return 1
        return 0
    
    for child in tree.get("children", []):
        count += count_mp4_in_tree(child)
    return count

if __name__ == "__main__":
    # Base directory
    base_dir = os.path.join("DataSets", "deepfakes")
    
    # Target folders specified in the request
    target_folders = [
        "source_videos_part_00",
        "source_videos_part_01",
        "source_videos_part_03",
        "source_videos_part_04",
        "manipulated_videos_part_00",
        "manipulated_videos_part_01",
        "manipulated_videos_part_02",
        "manipulated_videos_part_11"
    ]
    
    output_file = "dataset_hierarchy.json"
    
    print(f"Scanning directories in {base_dir}...")
    hierarchy_data = get_directory_hierarchy(base_dir, target_folders)
    
    # Generate report
    report = {
        "total_source_mp4": 0,
        "total_manipulated_mp4": 0,
        "details": {}
    }
    
    for folder_name, tree in hierarchy_data.items():
        mp4_count = count_mp4_in_tree(tree)
        report["details"][folder_name] = mp4_count
        
        if "source" in folder_name:
            report["total_source_mp4"] += mp4_count
        elif "manipulated" in folder_name:
            report["total_manipulated_mp4"] += mp4_count
            
    final_output = {
        "report": report,
        "hierarchy": hierarchy_data
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Hierarchy and report successfully saved to {output_file}")
