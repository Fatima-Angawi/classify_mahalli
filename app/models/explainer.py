import anthropic
class Explainer:
    def __init__(self):
        self.client = anthropic.Anthropic() # reads ANTHROPIC_API_KEY from env
    def explain(self, text: str, prob: float, tier: str) -> str:
        """Generate explanation for HUMAN_REVIEW listings only."""
        if tier != "HUMAN_REVIEW":
            return None # AUTO_REMOVE and CLEAR don't need explanation


        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
"role": "user",
"content": f"""You are a policy reviewer for Salla marketplace.
Analyze this product listing and explain in 2-3 sentences why it
may or may not violate policies around illegal streaming subscriptions,
shared accounts, game cheats, or unauthorized software resale.
Listing text: {text}
Scam probability: {prob:.0%}
Be specific. Mention exact words or patterns that are suspicious.
If it looks legitimate, say why."""
}]
)
        return response.content[0].text