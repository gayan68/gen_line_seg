from src.data.text.common_token_txt import BLANK_STR_TOKEN


class CharsetToken(object):
    """
    """

    def __init__(self, charset_file,  use_blank=False, ):
        self.charset_dictionary = {}
        self.charset_list = []
        self.char_number = 0

        if use_blank:
            self.charset_dictionary[BLANK_STR_TOKEN] = 0
            self.charset_list.append(BLANK_STR_TOKEN)
            self.char_number += 1

        with open(charset_file, mode='r', encoding="utf-8") as f:
            for line in f.readlines():
                if len(line) > 0:
                    c = line[:-1]
                    self.charset_dictionary[c] = self.char_number
                    self.charset_list.append(c)
                    self.char_number += 1

    def get_charset_dictionary(self):
        return self.charset_dictionary

    def get_charset_list(self):
        return self.charset_list

    def get_nb_char(self):
        return len(self.charset_list)
