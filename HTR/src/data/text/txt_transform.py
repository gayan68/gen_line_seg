def transcript_tranforme_base(dictionary, str_to_transform):
    """
    """

    labels = []

    for c in str_to_transform:
        if c not in dictionary:
            print("Text unknow char in dictionnary : " + str(c))
            print("Ignore")
            continue
            #return -1
        else:
            labels.append(dictionary.get(c))

    return labels


def transcript_tranform_token_split(dictionary, str_to_transform):
    """
    """

    labels = []

    token_str = str_to_transform.split(sep=' ')

    nb_item_unknown = 0
    for c in token_str:
        if c in dictionary:
             labels.append(dictionary.get(c))
        else:
            nb_item_unknown += 1
            # if len(c) > 0:
            #     for one_character in c:
            #         if one_character in dictionary:
            #             labels.append(dictionary.get(one_character))
            #         else:
            #             print("Text unknow char in dictionnary : " + str(one_character))

    if nb_item_unknown != 0:
        print("nb_item_unknown:" + str(nb_item_unknown))

    return labels


def convert_int_to_chars(indices, char_list):
    """
    """

    chars_sequence = ""

    for char_index in indices:
        try:
            c = char_list[char_index]

            chars_sequence += c
        except Exception as e:
            chars_sequence += "Error char index"

    return chars_sequence


def convert_int_to_chars_add_spaces(indices, char_list, space_token):
    """
    """

    chars_sequence = ""

    for char_index in indices:
        try:
            c = char_list[char_index]

            chars_sequence += c
            chars_sequence += space_token  # " "
        except Exception as e:
            chars_sequence += "Error char index"

    # remove last space
    if len(chars_sequence) > 0:
        size_space_token = len(space_token)
        chars_sequence = chars_sequence[:-size_space_token]

    return chars_sequence
