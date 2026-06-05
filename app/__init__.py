import os

print(
    "API KEY EXISTS:",
    bool(os.getenv("GEMINI_API_KEY"))
)