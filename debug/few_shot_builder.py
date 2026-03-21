# debug/few_shot_builder.py

import json

def get_few_shots_for_column(column_name):
    """
    Generate JSON array contoh input → golden output untuk kolom tertentu.
    Ini untuk few-shot learning di LLM.
    """

    # Contoh minimal untuk 3 kolom target
    few_shots = []

    if column_name == "Risk ID":
        few_shots = [
            {"input": "Row text with risk about data sharing", "output": "R1"},
            {"input": "Row text with ice data failure", "output": "R2"},
        ]
    elif column_name == "Risk Description":
        few_shots = [
            {"input": "Row text with risk about data sharing", "output": "Data sharing from BBSRI at risk"},
            {"input": "Row text with ice data failure", "output": "Ice data collection unsuccessful"},
        ]
    elif column_name == "Mitigating Action":
        few_shots = [
            {"input": "Row text with risk about data sharing", "output": "Facilitate conversations with Umaine on data sharing"},
            {"input": "Row text with ice data failure", "output": "Implement real-time data collection"},
        ]
    # Bisa ditambah untuk kolom lain
    else:
        few_shots = [
            {"input": "Row text example", "output": "DUMMY"}
        ]

    return json.dumps(few_shots, indent=2)