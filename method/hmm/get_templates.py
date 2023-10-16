import json

def get_templates(chords):
    """read from JSON file to get chord templates"""
    with open("./data/chord_templates.json", "r") as fp:
        templates_json = json.load(fp)
    templates = []

    for chord in chords:
        if chord in templates_json:
            templates.append(templates_json[chord])
        else:
            continue
            # "N": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "N1": [1,1,1,1,1,1,1,1,1,1,1,1]}

    return templates

def get_nested_circle_of_fifths():
    chords = [
        "C:maj", 
        "C#:maj",
        "D:maj",
        "D#:maj",
        "E:maj", 
        "F:maj", 
        "F#:maj", 
        "G:maj",
        "G#:maj", 
        "A:maj", 
        "A#:maj",
        "B:maj",
        "C:min", 
        "C#:min", 
        "D:min", 
        "D#:min", 
        "E:min", 
        "F:min", 
        "F#:min", 
        "G:min",
        "G#:min", 
        "A:min", 
        "A#:min",
        "B:min",
    ]
    nested_cof = [
        "C:maj",
        "E:min",
        "G:maj",
        "B:min",
        "D:maj",
        "F#:min",
        "A:maj",
        "C#:min",
        "E:maj",
        "G#:min",
        "B:maj",
        "D#:min",
        "F#:maj",
        "A#:min",
        "C#:maj",
        "F:min",
        "G#:maj",
        "C:min",
        "D#:maj",
        "G:min",
        "A#:maj",
        "D:min",
        "F:maj",
        "A:min",
    ]
    return chords, nested_cof