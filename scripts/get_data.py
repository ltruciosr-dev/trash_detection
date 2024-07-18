import os
import shutil
import numpy as np
import tqdm
import splitfolders

from pycocotools.coco import COCO

DATA_DIR = '../TACO/data'
CATEGORY_LABELS = {
    'Bottle': [4, 5, 6],
    'Bottle cap': [7, 8],
    'Cup': [21, 24, 20, 23],
    'Lid': [27, 28],
    'Other plastic': [29],
    'Plastic bag & wrapper': [36, 39],
    'Plastic container': [47],
    'Plastic glooves': [48],
    'Plastic utensils': [49],
    'Straw': [55, 56],
    'Can': [10, 12],
    'Styrofoam piece': [57],
    'Paper': [30, 32, 33],
    'Paper bag': [34, 35],
    'Carton': [14, 15, 16, 17, 18],
    'Broken glass': [9],
    'Glass jar': [26],
    'Aluminium foil': [0],
    'Blister pack': [2],
    'Scrap metal': [52],
    'Cigarette': [59]
}


if __name__ == "__main__":
    data_source = COCO(annotation_file=f'{DATA_DIR}/annotations.json')

    # Map from taco labels to our project labels
    mapping_labels = {}
    category_label = {}

    for i, (supercategory, labels) in enumerate(CATEGORY_LABELS.items()):
        category_label[supercategory] = i
        for label in labels:
            mapping_labels[label] = i

    # Create the temporal directories for the images
    save_image_path = 'tmp/images/'
    save_base_path  = 'tmp/labels/'
    
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_base_path, exist_ok=True)

    # Iterate over all the images
    img_ids = data_source.getImgIds()
    class_num = {}

    for img_id in tqdm.tqdm(img_ids, desc='iterate over taco images'):
        img_info = data_source.loadImgs(img_id)[0]
        save_name = img_info['file_name'].replace('/', '_')
        file_name = save_name.split('.')[0]
        height = img_info['height']
        width = img_info['width']
        
        # Check if image is downloaded
        img_path = f"{DATA_DIR}/{img_info['file_name']}"
        if not os.path.exists(img_path):
            continue
        save_path = save_base_path + file_name + '.txt'
        is_exist = False 
        with open(save_path, mode='w') as fp:
            annotation_id = data_source.getAnnIds(img_id)
            boxes = np.zeros((0, 5))
            if len(annotation_id) == 0:  # image without any annotations
                fp.write('')
                continue
            annotations = data_source.loadAnns(annotation_id)
            lines = ''
            for annotation in annotations:
                label = annotation['category_id']
                # Only download the images which the required categories
                if label in mapping_labels.keys():
                    is_exist = True
                    box = annotation['bbox']
                    if box[2] < 1 or box[3] < 1:
                        continue
                    # Mapping (top_x,top_y,width,height) => (cen_x,cen_y,width,height)
                    box[0] = round((box[0] + box[2] / 2) / width, 6)
                    box[1] = round((box[1] + box[3] / 2) / height, 6)
                    box[2] = round(box[2] / width, 6)
                    box[3] = round(box[3] / height, 6)
                    new_label = mapping_labels[label]
                    if new_label not in class_num.keys():
                        class_num[new_label] = 0
                    class_num[new_label] += 1
                    lines = lines + str(new_label)
                    for i in box:
                        lines += ' ' + str(i)
                    lines += '\n'
            fp.writelines(lines)
        # Store both the image and the label to the corresponding directories
        if is_exist:
            shutil.copy(f"{DATA_DIR}/{img_info['file_name']}", os.path.join(save_image_path, save_name))
        else:
            os.remove(save_path)

    # Split the folders into train and validation
    splitfolders.ratio('tmp', output="data", seed=1337, ratio=(.8, 0.2))

    # Remove the temporal directories
    shutil.rmtree('tmp')