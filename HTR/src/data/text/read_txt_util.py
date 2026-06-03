from enum import Enum

from src.data.text.filter_txt import filter_tag_clear_text, filter_tag_clear_plain_text
from src.data.text.txt_transform import transcript_tranform_token_split, transcript_tranforme_base, \
    convert_int_to_chars, convert_int_to_chars_add_spaces


class READ_TEXT_FORMAT(Enum):
    RAW = 1  # my word
    CLASSES_SPACED_WITH_SPACE = 2  # m y SPACE w o r d

    def __str__(self):
        return self.name


class SPACE_VALUE(Enum):
    RAW = 1
    TEXT = 2  # <SPACE>

    def __str__(self):
        return self.name


class FILTER_TXT(Enum):
    NO = 1
    CLEAR_TEXT = 2  # <SPACE>
    CLEAR_PLAIN_TEXT = 3

    def __str__(self):
        return self.name


def read_all_txt(path_file):
    content_str = ""

    with open(path_file, encoding="utf-8") as file:
        all_lines = file.readlines()

        for one_l in all_lines:
            one_l = one_l.replace("\n", "")

            content_str += one_l

    return content_str


class Text_Reader:
    def __init__(self, format: READ_TEXT_FORMAT, char_dict, add_space_before_after, space_value: SPACE_VALUE,
                 filter_config):

        self.char_dict = char_dict
        self.format = format
        self.add_space_before_after = add_space_before_after

        if filter_config == FILTER_TXT.NO:
            self.filter_fn = None
        elif filter_config == FILTER_TXT.CLEAR_TEXT:
            self.filter_fn = filter_tag_clear_text
        elif filter_config == FILTER_TXT.CLEAR_PLAIN_TEXT:
            self.filter_fn = filter_tag_clear_plain_text

        if space_value == SPACE_VALUE.RAW:
            self.space_str = " "
        elif space_value == SPACE_VALUE.TEXT:
            self.space_str = "<SPACE>"
        else:
            print("space value not recognize use default value")
            self.space_str = " "

    def read_text(self, x):
        text = read_all_txt(x)

        if self.filter_fn is not None:
            list_txt_filter = self.filter_fn(text)  # Return list

            text = ' '.join(list_txt_filter)

        if self.add_space_before_after == 1:
            if self.format == READ_TEXT_FORMAT.RAW:
                text = self.space_str + text + self.space_str
            elif self.format == READ_TEXT_FORMAT.CLASSES_SPACED_WITH_SPACE:
                text = self.space_str + " " + text + " " + self.space_str

        return text

    def read_text2(self, text):
        if self.filter_fn is not None:
            list_txt_filter = self.filter_fn(text)  # Return list

            text = ' '.join(list_txt_filter)

        if self.add_space_before_after == 1:
            if self.format == READ_TEXT_FORMAT.RAW:
                text = self.space_str + text + self.space_str
            elif self.format == READ_TEXT_FORMAT.CLASSES_SPACED_WITH_SPACE:
                text = self.space_str + " " + text + " " + self.space_str

        return text

    def transcript_txt_to_index(self, x):
        if self.format == READ_TEXT_FORMAT.RAW:
            return transcript_tranforme_base(self.char_dict, x)
        elif self.format == READ_TEXT_FORMAT.CLASSES_SPACED_WITH_SPACE:
            return transcript_tranform_token_split(self.char_dict, x)
        else:
            print("Transcript format not define.")

    def remove_space_before_after_one_item(self, txt):
        if self.add_space_before_after:
            if self.format == READ_TEXT_FORMAT.RAW:
                return txt.strip()
            elif self.format == READ_TEXT_FORMAT.CLASSES_SPACED_WITH_SPACE:
                len_space_str = len(self.space_str)

                if len(txt) > 2 * len_space_str:

                    txt_process = txt

                    # Case two succesives space
                    txt_process = txt_process.replace(self.space_str + " " + self.space_str, self.space_str)
                    nb_char = len(txt)
                    index_start_end_last_space = nb_char - len_space_str

                    if txt[:len_space_str] == self.space_str:
                        txt_process = txt_process[len_space_str + 1:]  # +1 == real space between classes
                    if txt[index_start_end_last_space:] == self.space_str:
                        txt_process = txt_process[:-len_space_str - 1]  # -1 == real space between classes

                    return txt_process
                else:
                    return txt
            else:
                print("format unknow remove space before after one item")

        else:
            return txt

    def ctc_best_path_one(self, index_class, char_list, token_blank):
        # Remove the duplicated characters index
        sequence_without_duplicates = []
        previous_index = -1
        for index in index_class:
            if index != previous_index:
                sequence_without_duplicates.append(index)
                previous_index = index

        # Remove the blanks
        sequence = []
        for index in sequence_without_duplicates:
            if index != token_blank:
                sequence.append(index)

        # Convert to characters
        char_sequence = "Not Convert"
        if self.format == READ_TEXT_FORMAT.RAW:
            char_sequence = convert_int_to_chars(sequence, char_list)
        elif self.format == READ_TEXT_FORMAT.CLASSES_SPACED_WITH_SPACE:
            char_sequence = convert_int_to_chars_add_spaces(sequence, char_list, space_token=" ")
        else:
            print("format unknow ctc_best_path_one")

        return char_sequence
