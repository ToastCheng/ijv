geometry = {
        "skin_thickness": 1,
        "fat_thickness": 1,
        "ijv_radius": 7.29,
        "ijv_depth": 15.25,
        "cca_radius": 2.475,
        "cca_depth": 25.11,
        "ijv_cca_distance": 6.48
    }

import json
with open("clinic_data/20190319_johnson.json", "w+") as f:
    json.dump(geometry, f, indent=4)