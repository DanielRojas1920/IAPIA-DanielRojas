import json
import os
# Paths
image_folder = 'train2017_Google'  # folder with selected images
input_json_path = 'annotations_trainval2017/annotations/instances_train2017.json'  # original
output_json_path = 'annotations_trainval2017/annotations/instances_train_subset.json'  # new

captions_input_json_path = 'annotations_trainval2017/annotations/captions_train2017.json'  # original
captions_output_json_path = 'annotations_trainval2017/annotations/captions_train_subset.json'  # new

# 1. Get list of file names without extension
image_files = set(os.listdir(image_folder))
image_names = set([f for f in image_files if f.endswith('.jpg')])
image_ids_keep = set()

# 2. Load JSON for object detection instances
with open(input_json_path, 'r') as f:
    coco_data = json.load(f)

# 3. Filter images
filtered_images = []
image_filename_to_id = {}

for img in coco_data['images']:
    if img['file_name'] in image_names:
        filtered_images.append(img)
        image_ids_keep.add(img['id'])
        image_filename_to_id[img['file_name']] = img['id']

# 4. Filter annotations
filtered_annotations = [
    ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids_keep
]

# 5. Filter categories (optional, but we keep it as is in most cases)
filtered_categories = coco_data['categories']

# 6. Save new JSON for instances
filtered_coco = {
    'images': filtered_images,
    'annotations': filtered_annotations,
    'categories': filtered_categories
}

with open(output_json_path, 'w') as f:
    json.dump(filtered_coco, f)

print(f"New annotation (object detection) saved at: {output_json_path}")
print(f"Total images: {len(filtered_images)}")
print(f"Total annotations: {len(filtered_annotations)}")

# -------------------------------------------
# Now, let's handle the annotations
# -------------------------------------------

# 7. Load JSON for annotations
with open(captions_input_json_path, 'r') as f:
    captions_data = json.load(f)

# 8. Filter captions based on selected images
filtered_captions = [
    caption for caption in captions_data['annotations'] if caption['image_id'] in image_ids_keep
]

# 9. Save new annotations file
filtered_captions_data = {
    'images': [img for img in coco_data['images'] if img['id'] in image_ids_keep],
    'annotations': filtered_captions
}

with open(captions_output_json_path, 'w') as f:
    json.dump(filtered_captions_data, f)

print(f"New captions saved at: {captions_output_json_path}")
print(f"Total captions: {len(filtered_captions)}")
