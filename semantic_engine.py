def group_by_zone(detections, img_width):
    zones = {"left": [], "center": [], "right": []}
    for _, row in detections.iterrows():
        x_center = (row['xmin'] + row['xmax']) / 2
        label = row['name']
        if x_center < img_width / 3:
            zones["left"].append(label)
        elif x_center < 2 * img_width / 3:
            zones["center"].append(label)
        else:
            zones["right"].append(label)
    return zones
