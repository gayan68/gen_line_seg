import os
import cv2

dst_dir = "Training"
ext = "jpg"

file_root = '../../DATASETS/READ_2016'
file_path = f"{file_root}/{dst_dir}"
img_dir = f'{file_path}/Images'
bb_dir = f'{file_path}/gt_boxes'
text_dir = f'{file_path}/gt_text'
dst_img_dir = f'{file_path}/image_lines'
dst_text_dir = f'{file_path}/text_lines'

print(file_path)
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(f'.{ext}')]

for idx, img_name in enumerate(img_files):
    img = cv2.imread(f"{img_dir}/{img_name}") 
    print(img.shape)
    base_name = os.path.splitext(img_name)[0]
    print(base_name)

    text_gt = []
    with open(f"{text_dir}/{base_name}.txt", "r") as file:
        for line in file:
            line = line.strip()
            text_gt.append(line)

    i=0
    with open(f"{bb_dir}/{base_name}.txt", "r") as file:
        # Loop through each line
        for line in file:
            # Strip removes newline characters
            line = line.strip()
            cords = line.split(" ")
            x1, y1, x2, y2 = map(int, cords)
            # print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

            cropped_img = img[x1:x2, y1:y2] 
            cv2.imwrite(f"{dst_img_dir}/{base_name}_{i}.JPG", cropped_img)

            print(text_gt[i])
            with open(f"{dst_text_dir}/{base_name}_{i}.txt", "w") as f:
                f.write(text_gt[i])
            i += 1
