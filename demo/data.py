from dataclasses import dataclass


@dataclass(frozen=True)
class Ticket:
    name: str
    report: str
    expected: str


QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this customer report?",
    "criteria": {
        "billing": "Charges, invoices, refunds, or payment problems",
        "technical": "Application bugs, failures, or account access problems",
        "other": "Neither billing nor technical support fits",
    },
}

SAMPLES = [
    Ticket("Duplicate charge", "I was charged twice for September. Please refund the duplicate payment.", "billing"),
    Ticket("Invoice missing", "My paid invoice is missing from the billing page.", "billing"),
    Ticket("Card declined", "Your checkout keeps declining my valid credit card.", "billing"),
    Ticket("Incorrect tax", "The sales tax on this invoice is wrong.", "billing"),
    Ticket("Settings crash", "The application crashes every time I open settings.", "technical"),
    Ticket("Login failure", "I cannot log in after resetting my password.", "technical"),
    Ticket("API error", "POST /orders returns 500 for every request today.", "technical"),
    Ticket("Blank screen", "The dashboard shows a blank screen in Firefox.", "technical"),
    Ticket("Docs typo", "The getting started guide has a typo in step three.", "other"),
    Ticket("Dark mode", "Please add a dark mode to the application.", "other"),
    Ticket("Partnership", "I'd like to discuss a marketing partnership.", "other"),
    Ticket("Thanks", "Everything works well. Thank you for the quick response.", "other"),
    Ticket("Negated refund", "I do not need a refund. The app simply will not open.", "technical"),
    Ticket("Negated bug", "The application works fine, but my invoice amount is incorrect.", "billing"),
]

KNOWLEDGE_BASE = {
    "billing": "Billing policy: Never promise a refund or change an invoice automatically. Verify account identity and payment details before assigning a human billing specialist.",
    "login": "Login troubleshooting: Ask for the exact error and timestamp. Check whether the reset link has expired, whether SSO is required, and whether cookies are blocked. Escalate lockouts to identity operations.",
    "api": "API incident checklist: Capture the request ID, endpoint, timestamp, status code, and recent deployment. Inspect service health and error logs. Escalate persistent 5xx to on-call engineering.",
    "ui": "UI troubleshooting: Record browser/version, route, console errors, and reproduction steps. Try a clean profile and check recent frontend releases. Escalate reproducible crashes.",
}
