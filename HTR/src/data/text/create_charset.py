import glob


def create_charset(dir_data, path_save):
    """
    Read all label ground truth and create a file with all characters present
    """
    files = glob.glob(dir_data + '/**/*.txt', recursive=True)

    full_text = ""

    for one_file_label in files:
        label = ""
        with open(one_file_label, "r", encoding="utf-8") as file:
            label = file.readline()

        full_text += label

    charset = set(full_text)
    charset = sorted(charset)

    with open(path_save, 'w', encoding="utf-8") as file:
        for one_char in charset:
            file.write(one_char)
            file.write("\n")


