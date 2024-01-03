def change_color(button, intvar):
    current_color = button["bg"]

    if current_color == "white":
        new_color = "orange"
        intvar.set(1)

    else:
        new_color = "white"
        intvar.set(0)

    button["bg"] = new_color
