"""Step 8 (decline letter): draft the letter sent to an applicant from the decision and its reasons.

A language-model letter is a **draft for the analyst to check and edit**, never sent automatically.
Tested on llama3.2:3b (28 Oct): the guardrails below catch hard violations (one in six letters
judged a reason, "longer than we consider suitable", and fell back to the template), but softer
wording ("might not be the most suitable option") still gets through. A human stays in the loop.

Course reference: Code Bank W06-03 (one function over several LLM providers, with a fallback).

    letter = draft_letter("decline", result["reasons"])
    letter["text"]      # the letter
    letter["backend"]   # "template" or "ollama": which one actually wrote it
    letter["warning"]   # None, or why the template was used instead of the chosen backend

Backends (LLM_BACKEND in .env):
- "template": fixed wording with the reasons filled in. No language model: deterministic, free,
  works offline, and the wording can be signed off by compliance once. The default, and what the
  public demo uses.
- "ollama": a small local language model writes a more personal letter. It runs on our own
  machine (closed environment), so no customer data is sent to an outside AI provider.

Guardrails around the language model:
- Input: the prompt only gets the decision and the reason phrases, which must come from the fixed
  list in src/explain.py. Never the PD, the applicant's inputs or protected attributes, and no
  free text that could carry instructions (prompt injection).
- A timeout, and any error falls back to the template, with a warning the app shows.
- Output: the letter is rejected (and the template used) if it is empty or too long, contains a
  number or percentage (the model must not invent scores, rates or deadlines), mentions a
  protected attribute (age, sex, family status, race, religion...: declining someone for these
  reasons would be unlawful discrimination, so the letter must never suggest it), judges a reason
  ("too long", "did not meet"), or leaves out the right to a review by a credit officer.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from src.config import LENDER_NAME, LLM_BACKEND, LLM_TIMEOUT_SECONDS, OLLAMA_MODEL, OLLAMA_URL
from src.explain import REASON_PHRASES

BACKENDS = ("template", "ollama")
MAX_LETTER_CHARS = 2500
# Words a letter must never contain: protected attributes under Australian anti-discrimination law.
PROTECTED_WORDS = re.compile(
    r"\b(age|aged|old|young|gender|sex|male|female|married|marital|single|divorced|widow\w*|pregnan\w*|"
    r"race|racial|ethnic\w*|religio\w*|disabilit\w*|health)\b",
    re.IGNORECASE,
)
# A reason names a factor, not a verdict: several effects are U-shaped or reversed (longer loans
# default less in this data), so "your loan is too long" could be false for this applicant.
DIRECTION_WORDS = re.compile(
    r"\b(too|higher|lower|longer|shorter|greater|insufficient|poor|did not meet|didn't meet|does not meet)\b",
    re.IGNORECASE,
)
SIGN_OFF = f"Yours sincerely,\nCredit Assessment Team\n{LENDER_NAME}"


# ---- Template (always available) ---------------------------------------------------
def template_letter(decision: str, reasons: list[str]) -> str:
    """Fixed, reviewable wording with the reasons filled in."""
    if decision == "refer":
        return (
            "Dear Applicant,\n\n"
            "Thank you for your recent loan application. Before we can make a decision, it needs a closer "
            "look by one of our credit officers. They may contact you to ask for more information about your "
            "income, employment or the loan you have asked for.\n\n"
            "You don't need to do anything now. We will let you know the outcome as soon as the review is "
            "complete.\n\n" + SIGN_OFF
        )
    if reasons:
        factors = "Our decision was based mainly on the following factors:\n" + "".join(
            f"- {r}\n" for r in reasons
        )
    else:
        factors = "Our decision was based on our overall assessment of your application.\n"
    return (
        "Dear Applicant,\n\n"
        "Thank you for your recent loan application. After careful review, we are unable to approve it at "
        "this time.\n\n"
        + factors
        + "\nThis decision is based on the information in your application and the credit information "
        "available to us. It is not a judgement of you as a person, and it does not stop you from applying "
        "again.\n\n"
        "You can ask us which credit information we used, and you can ask for this decision to be reviewed "
        "by one of our credit officers. If your circumstances change, or you would like to apply for a "
        "smaller amount, we would be glad to consider a new application.\n\n" + SIGN_OFF
    )


# ---- Language model (Ollama, local) ------------------------------------------------------
def build_prompt(decision: str, reasons: list[str]) -> str:
    """The only text the language model sees: the decision and the reason phrases, nothing else."""
    if decision == "decline":
        task = (
            "Write a letter telling the applicant that their loan application was declined, and which "
            "factors the decision was mainly based on."
        )
    else:
        task = (
            "Write a letter telling the applicant that no decision has been made yet: a credit officer will "
            "review their application and may contact them for more information."
        )
        reasons = []  # nothing decided yet, so the model isn't given reasons it could present as final
    listed = "\n".join(f"- {r}" for r in reasons) or "- Overall assessment of the application"
    return (
        "You write short, clear and polite letters to loan applicants for a consumer lender.\n"
        f"{task}\n\n"
        "Rules:\n"
        "- Name each reason below as a factor the assessment looked at, in plain, friendly English. Do not "
        "say whether it was too high, too low, too long or too short, or that it did not meet requirements: "
        "the reasons say what mattered most, not a verdict on each one. Do not add any other reasons.\n"
        "- Do not add facts about interest, fees, credit bureaus or how the lender works, and do not say the "
        "decision is not about their credit.\n"
        "- Do not use any numbers, percentages, scores, rates or deadlines.\n"
        "- Do not mention or guess anything personal about the applicant, such as their age, sex, family, "
        "health, background or beliefs.\n"
        "- Do not promise that a future application will be approved.\n"
        "- Say that they can ask which credit information was used and ask for a review by a credit officer.\n"
        "- Keep it under 150 words. Start with 'Dear Applicant,' and end with exactly:\n"
        f"{SIGN_OFF}\n\n"
        f"Reasons:\n{listed}\n"
    )


def _call_ollama(prompt: str) -> str:
    """One request to the local Ollama server (standard library only, no extra package)."""
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    ).encode()
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
        return json.loads(response.read())["response"].strip()


def check_letter(text: str) -> str | None:
    """Why a generated letter can't be sent (None if it passes the output guardrails)."""
    if not text.strip():
        return "the letter was empty"
    if len(text) > MAX_LETTER_CHARS:
        return f"the letter was too long ({len(text):,} characters)"
    if re.search(r"\d|%", text):
        return "the letter contained a number or percentage"
    if match := PROTECTED_WORDS.search(text):
        return f"the letter mentioned a protected attribute ('{match.group(0)}')"
    if match := DIRECTION_WORDS.search(text):
        return f"the letter judged a reason ('{match.group(0)}'), which may not be true for this applicant"
    if "credit officer" not in text.lower():
        return "the letter didn't mention the right to a review by a credit officer"
    return None


# ---- One entry point for the app ------------------------------------------------------------
def draft_letter(decision: str, reasons: list[str], backend: str = LLM_BACKEND) -> dict:
    """Draft a decline or refer letter. Never raises because a backend failed: it falls back to the
    template and says why in "warning", so the app always has a letter to show."""
    if decision not in ("decline", "refer"):
        raise ValueError(f"Letters are only drafted for 'decline' or 'refer', not '{decision}'")
    unknown = [r for r in reasons if r not in REASON_PHRASES.values()]
    if unknown:
        raise ValueError(f"Unknown reasons (only phrases from src/explain.py are allowed): {unknown}")

    if backend == "template":
        return {"text": template_letter(decision, reasons), "backend": "template", "warning": None}
    if backend not in BACKENDS:
        problem = f"unknown LLM_BACKEND '{backend}' (options: {', '.join(BACKENDS)})"
    else:
        try:
            text = _call_ollama(build_prompt(decision, reasons))
            problem = check_letter(text)
            if problem is None:
                return {"text": text, "backend": "ollama", "warning": None}
        except TimeoutError:
            problem = f"the local model took longer than {LLM_TIMEOUT_SECONDS} s"
        except urllib.error.HTTPError as e:  # before URLError, which it extends
            if e.code == 404:
                problem = (
                    f"the model '{OLLAMA_MODEL}' isn't installed in Ollama (run: ollama pull {OLLAMA_MODEL})"
                )
            else:
                problem = f"Ollama returned an error ({e.code} {e.reason})"
        except urllib.error.URLError as e:
            problem = f"couldn't reach Ollama at {OLLAMA_URL} ({e.reason}); is it running?"
        except (KeyError, ValueError) as e:  # unexpected or non-JSON reply
            problem = f"unexpected reply from Ollama ({e})"
    return {
        "text": template_letter(decision, reasons),
        "backend": "template",
        "warning": f"Standard letter used because {problem}.",
    }
