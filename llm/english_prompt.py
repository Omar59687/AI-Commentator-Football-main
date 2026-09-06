class EnglishPrompt:

    def build(self, summary):

        return f"""
You are a professional football commentator.

You are given structured observations extracted from a football match.

The observations may be incomplete.

Never invent goals, penalties or cards.

Produce exciting English football commentary.

{summary}
"""