import xml.etree.ElementTree as ET
import shutil

split_file =  "validation.uttlist"
dst_dir = "Validation"

splitfile_root = '../../DATASETS/IAM/Aachen_Splits'
split_file_path = f"{splitfile_root}/{split_file}"
xml_root = '../../DATASETS/IAM/xml'
src_root = '../../DATASETS/IAM/lines'
dst_root = '../../DATASETS/IAM/line_splits'
dst_path = f'{dst_root}/{dst_dir}'

print(split_file_path)
with open(split_file_path, "r") as file:
    # Loop through each line
    for line in file:
        # Strip removes newline characters
        line = line.strip()
        print(line)



        tree = ET.parse(f"{xml_root}/{line}.xml")
        root = tree.getroot()

        # Iterate over all <line> elements
        for line in root.iter("line"):
            segmentation = line.attrib.get("segmentation")
            line_id = line.attrib.get("id")
            # line_id_lst = line_id.split("-")
            # print(f"Line ID: {line_id}, Segmentation: {segmentation}")


            # if segmentation == "ok":
            #     src_path = f"{src_root}/{line_id_lst[0]}/{line_id_lst[0]}-{line_id_lst[1]}/{line_id}.png"
            #     shutil.copy(src_path, f"{dst_path}/{line_id}.png")
            #     print("copy")

            text = line.attrib.get("text")
            with open(f"{dst_path}/gt_text/{line_id}.txt", "a") as f:
                f.write(text)
