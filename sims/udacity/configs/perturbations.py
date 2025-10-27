SEVERITIES = [1, 2, 3, 4]
EPISODES = 1 # repeats per (perturb, severity) per road

# Flat list (used if CHUNKS == [])
LIST = [
    # Weather (dynamic/statics that are commonly used & fast enough)
    "fog_filter",
    "dynamic_rain_filter",
    "dynamic_snow_filter",
    "dynamic_sun_filter",
    "dynamic_lightning_filter",
    "dynamic_smoke_filter",
    # Statics (lighter than dynamics; load only if needed)
    "static_rain_filter",
    "static_snow_filter",
    "static_sun_filter",
    "static_lightning_filter",
    "static_smoke_filter",
    "frost_filter",

    # Blur / focus (no zoom blur)
    "motion_blur",
    "defocus_blur",
    "gaussian_blur",

    # Noise & compression
    "gaussian_noise",
    "poisson_noise",
    "impulse_noise",
    "speckle_noise_filter",
    "jpeg_filter",
    "pixelate",

    # Color / tone
    "increase_brightness",
    "contrast",
    "white_balance_filter",
    "saturation_filter",
    "saturation_decrease_filter",
    "posterize_filter",
    "histogram_equalisation",
    "grayscale_filter",
    "false_color_filter",
    "phase_scrambling",

    # Distortions & graphics
    "elastic",
    "cutout_filter",
    "sample_pairing_filter",
    "splatter_mapping",
    "dotted_lines_mapping",
    "zigzag_mapping",
    "canny_edges_mapping",

    # Geometric (safe subset used in paper for LK/ACC)
    "translate_image",
    "scale_image",

    # Overlays
    "reflection_filter",
    "dynamic_object_overlay",
    "static_object_overlay",
]

TRY = [
    "dynamic_rain_filter",
    "high_pass_filter", # if available in build
    "low_pass_filter", # if available in build
]


# Grouped for asset reuse; set CHUNKS = [] to fall back to LIST
CHUNKS = [
    # Weather families
    ["dynamic_rain_filter"],
    ["dynamic_snow_filter", "frost_filter", "fog_filter"],
    ["dynamic_sun_filter"],
    ["dynamic_lightning_filter"],
    ["dynamic_smoke_filter"],

    # Blur / focus
    ["motion_blur", "defocus_blur", "gaussian_blur", "low_pass_filter"],

    # Noise & compression
    ["gaussian_noise", "poisson_noise", "impulse_noise", "speckle_noise_filter", "jpeg_filter", "pixelate"],

    # Color / tone
    [
        "increase_brightness","contrast","white_balance_filter",
        "saturation_filter","saturation_decrease_filter","posterize_filter",
        "histogram_equalisation","grayscale_filter","false_color_filter",
        "high_pass_filter","low_pass_filter","phase_scrambling"
    ],

    # Distortions & graphics
    ["elastic","cutout_filter","sample_pairing_filter"],
    ["splatter_mapping","dotted_lines_mapping","zigzag_mapping","canny_edges_mapping"],

    # Geometry (safe subset)
    ["translate_image","scale_image"],

    # Overlays (may load bitmap assets)
    ["dynamic_object_overlay","static_object_overlay","reflection_filter"],
]



ALL_PERTURBATIONS = [
    # Weather (dynamic + static)
    "fog_filter",
    "dynamic_rain_filter", "static_rain_filter",
    "dynamic_raindrop_filter",  # heavy mask load; keep here for completeness
    "dynamic_snow_filter", "static_snow_filter", "frost_filter",
    "dynamic_sun_filter", "static_sun_filter",
    "dynamic_lightning_filter", "static_lightning_filter",
    "dynamic_smoke_filter", "static_smoke_filter",

    # Blur / focus (no zoom blur)
    "defocus_blur", "motion_blur", "gaussian_blur", "low_pass_filter",

    # Noise & compression
    "gaussian_noise", "poisson_noise", "impulse_noise", "speckle_noise_filter",
    "jpeg_filter", "pixelate",

    # Distortions
    "elastic", "sample_pairing_filter",

    # Graphics / occlusions
    "cutout_filter", "splatter_mapping", "dotted_lines_mapping", "zigzag_mapping", "canny_edges_mapping",

    # Color / tone
    "increase_brightness", "contrast", "white_balance_filter",
    "saturation_filter", "saturation_decrease_filter",
    "posterize_filter", "histogram_equalisation",
    "grayscale_filter", "false_color_filter",
    "high_pass_filter", "low_pass_filter", "phase_scrambling",

    # Geometry (safe subset evaluated for LK/ACC)
    "translate_image", "scale_image",

    # Overlays
    "reflection_filter",
    "dynamic_object_overlay", "static_object_overlay",

    # --- Explicitly excluded from ALL (paper reasons) ---
    # "zoom_blur", # too slow for real-time
    # "shear_image", # invalid semantics
    # "rotate_image", # invalid semantics
    # "cycle_consistent", # generative domain shift (unrealistic)
    # "candy","la_muse","mosaic","feathers","the_scream","udnie", # style-transfer
]
