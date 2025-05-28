def is_safe_to_cross(zones, vehicle_movements, crosswalk_occupied):
    """
    zones: dict {left:[], center:[], right:[]}
    vehicle_movements: dict {vehicle_id: movement_bool}  # True if moving
    crosswalk_occupied: bool, True if person detected in crosswalk zone
    """

    dangerous_objects = {"car", "bus", "truck", "motorcycle"}

    # 1. If crosswalk not detected or no pedestrian crossing object, not safe
    if "pedestrian crossing" not in [obj.lower() for obj in zones["center"]]:
        return False

    # 2. If people are crossing in crosswalk, allow crossing only if no moving vehicles
    if crosswalk_occupied:
        # Check if any vehicle is moving
        if any(vehicle_movements.values()):
            return False
        else:
            return True

    # 3. If crosswalk empty, allow crossing only if no vehicles at all or all stationary
    vehicles_present = False
    for zone in ["left", "center", "right"]:
        for obj in zones[zone]:
            if obj.lower() in dangerous_objects:
                vehicles_present = True
                # Check movement for this vehicle id? (you must track vehicle ids)
                # For now, assume moving if any vehicle moves
                if any(vehicle_movements.values()):
                    return False

    if not vehicles_present:
        return True

    # Default deny crossing
    return False
