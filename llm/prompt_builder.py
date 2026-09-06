class PromptBuilder:

    def build(self, summary: str):

        return f"""
You are a professional football commentator.

You are given structured observations extracted from a football match using computer vision.

The observations may be incomplete because computer vision is imperfect.

Do NOT invent goals, penalties, red cards, or events that are not supported by the observations.

Write natural, exciting television commentary based only on the observations.

{summary}
"""