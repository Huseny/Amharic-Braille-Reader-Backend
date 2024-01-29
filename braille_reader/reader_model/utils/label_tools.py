"""
Utilities to handle braille labels in various formats:
    int_label: label as int [0..63]
    label010: label as str of six 0 and 1s: '010101' etc.
    label123: label as str like '246'
    human_labels: labels in a manual annotation
"""

v = [1, 2, 4, 8, 16, 32]


def validate_int(int_label):
    """
    Validate int_label is in [0..63]
    Raise exception otherwise
    """
    assert isinstance(int_label, int)
    assert int_label >= 0 and int_label < 64, "Ошибочная метка: " + str(int_label)


def label010_to_int(label010):
    """
    Convert label in label010 format to int_label
    """
    r = sum([v[i] for i in range(6) if label010[i] == "1"])
    validate_int(r)
    return r


def label_vflip(int_lbl):
    """
    convert int_label in case of vertical flip
    """
    validate_int(int_lbl)
    return (
        ((int_lbl & (1 + 8)) << 2) + ((int_lbl & (4 + 32)) >> 2) + (int_lbl & (2 + 16))
    )


def label_hflip(int_lbl):
    """
    convert int_label in case of horizontal flip
    """
    validate_int(int_lbl)
    return ((int_lbl & (1 + 2 + 4)) << 3) + ((int_lbl & (8 + 16 + 32)) >> 3)


def int_to_label010(int_lbl):
    int_lbl = int(int_lbl)
    r = ""
    for i in range(6):
        r += "1" if int_lbl & v[i] else "0"
    return r


def int_to_unicode(int_lbl):
    return chr(0x2800 + int_lbl)