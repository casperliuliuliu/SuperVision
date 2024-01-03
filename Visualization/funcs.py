def change_color(button, BooleanVar):
    current_color = button["bg"]

    if current_color == "white":
        new_color = "orange"
        BooleanVar.set(True)

    else:
        new_color = "white"
        BooleanVar.set(False)

    button["bg"] = new_color
