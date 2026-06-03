import string


def filter_tag_clear_text(txt, clear_txt="<CLEARTEXT", space_str="<SPACE>"):
    # filter <CLEARTEXT zy:>  -> zy:
    # warning cleartext can contain space: <CLEARTEXT an 25.>
    txt_post_process = []

    current_class = ""

    i_c = 0

    while i_c < len(txt):
        if current_class == clear_txt:
            i_c += 1                    # +1 -> move from the first space
            i_c_search_clear_txt = i_c
            clear_txt_value = ""

            while txt[i_c_search_clear_txt] != ">":
                clear_txt_value += txt[i_c_search_clear_txt]
                i_c_search_clear_txt += 1
                i_c += 1

            # split text
            for one_c in clear_txt_value:
                if one_c == " 0" or one_c == " 6":
                    print("check sample")
                if one_c == " ":
                    txt_post_process.append(space_str)
                else:
                    txt_post_process.append(one_c)

            current_class = ""
        else:
            if txt[i_c] == " " and len(current_class) > 0:
                txt_post_process.append(current_class)
                current_class = ""
            elif txt[i_c] != " ":
                current_class += txt[i_c]

        i_c += 1

    # Add last
    if current_class != "":
        txt_post_process.append(current_class)

    return txt_post_process


def filter_tag_clear_plain_text(txt, clear_txt_1="<CLEARTEXT", clear_txt_2="<PLAINTEXT", space_str="<SPACE>"):
    # filter <CLEARTEXT zy:>  -> zy:
    # warning cleartext can contain space: <CLEARTEXT an 25.>
    txt_post_process = []

    current_class = ""

    i_c = 0

    while i_c < len(txt):
        if current_class == clear_txt_1 or current_class == clear_txt_2:
            i_c += 1                    # +1 -> move from the first space
            i_c_search_clear_txt = i_c
            clear_txt_value = ""

            while txt[i_c_search_clear_txt] != ">":
                clear_txt_value += txt[i_c_search_clear_txt]
                i_c_search_clear_txt += 1
                i_c += 1

            # split text
            for one_c in clear_txt_value:
                if one_c == " ":
                    txt_post_process.append(space_str)
                else:
                    txt_post_process.append(one_c)

            current_class = ""
        else:
            if txt[i_c] == " " and len(current_class) > 0:
                txt_post_process.append(current_class)
                current_class = ""
            elif txt[i_c] != " ":
                current_class += txt[i_c]

        i_c += 1

    # Add last
    if current_class != "":
        txt_post_process.append(current_class)

    return txt_post_process


def filter_index_letter(ind, char_dict):
    ind_letter = []

    for l in string.ascii_lowercase:
        if l in char_dict:
            ind_letter.append(char_dict[l])
    for l in string.ascii_uppercase:
        if l in char_dict:
            ind_letter.append(char_dict[l])

    ind_filtered = []

    for one_ind in ind:
        if one_ind not in ind_letter:
            ind_filtered.append(one_ind)

    return ind_filtered
