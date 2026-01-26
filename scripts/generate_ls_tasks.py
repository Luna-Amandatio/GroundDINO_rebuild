#!/usr/bin/env python3
"""Generate Label Studio task import JSON from dataset directory.

This script walks through the /data/datasets directory and creates a Label Studio
compatible task list with proper image paths and metadata.

Usage:
    python3 generate_ls_tasks.py

Output:
    Creates tasks.json ready for import into Label Studio
"""

import os
import json

# Root dataset directory (as mounted in Docker containers)
DATASET_ROOT = "/data/datasets"

# Output tasks file
OUTPUT_JSON = "tasks.json"

# Image extensions to include
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def main():
    """Generate Label Studio tasks from dataset directory."""

    tasks = []

    # Check if dataset root exists
    if not os.path.isdir(DATASET_ROOT):
        print(f"❌ Error: Dataset directory not found: {DATASET_ROOT}")
        print("   Make sure /data/datasets is mounted and accessible.")
        return

    # List all category folders
    categories = sorted(os.listdir(DATASET_ROOT))

    if not categories:
        print(f"❌ Error: No subdirectories found in {DATASET_ROOT}")
        return

    print(f"📁 Found {len(categories)} categories: {', '.join(categories)}\n")

    for category in categories:
        category_path = os.path.join(DATASET_ROOT, category)

        # Skip non-folders
        if not os.path.isdir(category_path):
            print(f"⏭️  Skipping non-directory: {category}")
            continue

        # Determine metadata
        is_field = category.lower() == "field"

        # Count images in this category
        image_count = 0

        # Walk through images
        for filename in sorted(os.listdir(category_path)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            # Absolute path as it appears in the container
            absolute_path = os.path.join(category_path, filename)

            # Label Studio local file serving format
            # Format: /data/local-files/?d=<absolute-path>
            # The path should be relative to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (/data)
            # So we strip /data/ prefix and use the rest
            relative_to_data_root = absolute_path.replace("/data/", "")
            image_url = f"/data/local-files/?d={relative_to_data_root}"

            task = {
                "data": {
                    "image": image_url,
                    "category": category,
                    "filename": filename,
                    "environment": {"field": is_field},
                }
            }

            tasks.append(task)
            image_count += 1

        print(f"  ✓ {category}: {image_count} images")

    # Write JSON file
    print(f"\n📝 Writing {len(tasks)} tasks to {OUTPUT_JSON}...")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"✅ Created {OUTPUT_JSON} with {len(tasks)} tasks")
    print("\n📋 Import instructions:")
    print("   1. Open Label Studio: http://localhost:8080")
    print("   2. Create or open a project")
    print("   3. Go to Import tab")
    print(f"   4. Upload {OUTPUT_JSON}")
    print("   5. Click Import")

    # Print sample task for verification
    if tasks:
        print("\n🔍 Sample task (first image):")
        print(json.dumps(tasks[0], indent=2))


if __name__ == "__main__":
    main()
