"""Shared schema constants."""

VISION_PROMPT_VERSION = "vision-v1"
AUDIT_PROMPT_VERSION = "audit-v1"
TRACE_SCHEMA_VERSION = "trace-v1"
VISION_SCHEMA_VERSION = "vision-schema-v1"
AUDIT_SCHEMA_VERSION = "audit-schema-v1"

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUSES = ("supported", "contradicted", "not_enough_information")

ISSUE_TYPES = (
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
)

OBJECT_PARTS = (
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
)

RISK_FLAGS = (
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
)

RISK_FLAG_PRIORITY = (
    "wrong_object",
    "wrong_object_part",
    "claim_mismatch",
    "damage_not_visible",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
)

SEVERITIES = ("none", "low", "medium", "high", "unknown")
CLAIM_OBJECTS = ("car", "laptop", "package")
