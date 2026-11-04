"""Station 4 (Implementation). Translate between the analyst's form and the model's input fields.

The model uses Home Credit's field names and units, e.g. DAYS_EMPLOYED = -2,190 (days, counted
backwards from the application) or 365243 for "not employed". An analyst thinks in years, so the
form asks for years and this module converts:

    model_fields = to_model_inputs(form_values)     # form -> predict()
    form_values = to_form_values(model_fields)      # example applicant (model fields) -> form

FIELDS holds the label and help text of every input, so the form and the error messages use the
same wording. Keeping this out of app.py keeps the UI thin and the conversion testable.
"""

from __future__ import annotations

import math

from src.config import DAYS_EMPLOYED_SENTINEL, RAW_INPUTS

DAYS_PER_YEAR = 365.25
# Model fields entered in years on the form: model field -> form field
YEAR_FIELDS = {
    "DAYS_EMPLOYED": "YEARS_EMPLOYED",
    "DAYS_REGISTRATION": "YEARS_SINCE_REGISTRATION",
    "DAYS_ID_PUBLISH": "YEARS_SINCE_ID",
}
NOT_EMPLOYED = "NOT_EMPLOYED"  # form checkbox; the model gets DAYS_EMPLOYED = 365243
EXT_SCORES = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
# Form checkbox per external score: ticked means the source has no score (the model gets None)
NO_SCORE = {field: f"{field}_NONE" for field in EXT_SCORES}
_SCORE_HELP = (
    "A credit score from an external data provider such as a credit bureau, scaled from 0 (highest risk) "
    "to 1 (lowest risk). It comes from the credit check the lender runs with the applicant's consent, not "
    "from the applicant. Home Credit anonymised its three providers, so we can't name them; an Australian "
    "bureau score (e.g. 0 to 1,200) would be divided by its maximum. These are the model's strongest inputs."
)

# Form field -> (label, help text). Every model input appears once, in its form units.
FIELDS = {
    # Loan
    "NAME_CONTRACT_TYPE": ("Loan type", "Cash loan (fixed amount) or revolving loan (credit line)."),
    "AMT_CREDIT": (
        "Loan amount ($)",
        "Amount to be lent. Also the exposure at default (EAD) in the profit calculation.",
    ),
    "AMT_ANNUITY": (
        "Regular repayment ($ per instalment)",
        (
            "Amount of each repayment. Loan amount ÷ repayment = number of instalments: past approved loans "
            "needed 8 to 45, and the lending policy declines anything over 48."
        ),
    ),
    "AMT_GOODS_PRICE": (
        "Price of goods financed ($)",
        "For consumer loans: the price of what the loan pays for. Leave blank for cash loans with no goods.",
    ),
    # Income and employment
    "AMT_INCOME_TOTAL": (
        "Annual income ($)",
        (
            "Total yearly income before tax, as stated and verified on the application. The lending policy "
            "declines a repayment larger than income or a loan more than 30 times income."
        ),
    ),
    "NAME_INCOME_TYPE": ("Income type", "Main source of income."),
    NOT_EMPLOYED: ("Not employed (e.g. pensioner)", "Tick if the applicant has no current job."),
    "YEARS_EMPLOYED": ("Years in current job", "Ignored if 'Not employed' is ticked."),
    "OCCUPATION_TYPE": ("Occupation", "Choose 'Unknown / not given' if the applicant didn't state it."),
    "ORGANIZATION_TYPE": (
        "Type of employer",
        (
            "The employer's industry as Home Credit coded it. 'Business Entity Type 1-3' are its codes for "
            "private companies of different kinds; 'XNA' means no employer (e.g. pensioners)."
        ),
    ),
    "NAME_EDUCATION_TYPE": ("Education", "Highest level completed."),
    # Credit history
    "EXT_SOURCE_1": ("External credit score 1 (0 to 1)", _SCORE_HELP),
    "EXT_SOURCE_1_NONE": (
        "No score from source 1",
        (
            "Tick if this provider has no score for the applicant. Common: 56% of past applicants had no "
            "score 1. The model treats a missing score as information (a thin credit file), not as zero."
        ),
    ),
    "EXT_SOURCE_2": ("External credit score 2 (0 to 1)", _SCORE_HELP),
    "EXT_SOURCE_2_NONE": (
        "No score from source 2",
        (
            "Tick if this provider has no score for the applicant. Rare: almost every past applicant had a "
            "score 2. The model treats a missing score as information (a thin credit file), not as zero."
        ),
    ),
    "EXT_SOURCE_3": ("External credit score 3 (0 to 1)", _SCORE_HELP),
    "EXT_SOURCE_3_NONE": (
        "No score from source 3",
        (
            "Tick if this provider has no score for the applicant. Common: 20% of past applicants had no "
            "score 3. The model treats a missing score as information (a thin credit file), not as zero."
        ),
    ),
    "AMT_REQ_CREDIT_BUREAU_YEAR": (
        "Credit bureau enquiries (last year)",
        (
            "From the credit report: how many times lenders checked the applicant's credit in the past 12 months. "
            "Many recent checks can mean the applicant is seeking credit in several places."
        ),
    ),
    # Household and stability
    "CNT_CHILDREN": ("Children", "Number of dependent children."),
    "CNT_FAM_MEMBERS": ("Household size", "Number of people in the household, including the applicant."),
    "NAME_HOUSING_TYPE": ("Housing", "Current housing situation."),
    "FLAG_OWN_REALTY": ("Owns property", "Y or N."),
    "FLAG_OWN_CAR": ("Owns a car", "Y or N."),
    "OWN_CAR_AGE": ("Car age (years)", "Ignored when 'Owns a car' is No."),
    "REGION_RATING_CLIENT_W_CITY": (
        "Region risk rating (lender's own)",
        (
            "The lender's risk rating of the area the applicant lives in, taking the city into account: 1 = "
            "lowest risk, 3 = highest (past default rates 4.8%, 7.9% and 11.4%). The applicant doesn't know it: "
            "the lender looks it up from the address. If unsure, use 2, which covers three in four applicants."
        ),
    ),
    "YEARS_SINCE_REGISTRATION": (
        "Years since registration details changed",
        (
            "From the applicant's records: years since they last changed their official registration details, "
            "such as their registered address."
        ),
    ),
    "YEARS_SINCE_ID": (
        "Years since ID document was renewed",
        "From the ID checked at application: years since the current identity document was issued.",
    ),
}


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value == ""


def to_model_inputs(form: dict) -> dict:
    """Form values (years, a 'not employed' tick) -> the model's input fields (negative days, codes)."""
    model = {f: (None if _blank(form.get(f)) else form.get(f)) for f in RAW_INPUTS if f not in YEAR_FIELDS}
    for model_field, form_field in YEAR_FIELDS.items():
        years = form.get(form_field)
        model[model_field] = None if _blank(years) else -round(float(years) * DAYS_PER_YEAR)
    if form.get(NOT_EMPLOYED):
        model["DAYS_EMPLOYED"] = DAYS_EMPLOYED_SENTINEL
    for field, no_score in NO_SCORE.items():
        if form.get(no_score):
            model[field] = None  # no score from this provider: the model's "missing" path (thin file)
    if form.get("FLAG_OWN_CAR") == "N":
        model["OWN_CAR_AGE"] = None  # no car, no car age (as in the training data)
    return model


def to_form_values(model: dict) -> dict:
    """The model's input fields -> form values, e.g. to load an example applicant into the form."""
    form = {f: (None if _blank(model.get(f)) else model.get(f)) for f in RAW_INPUTS if f not in YEAR_FIELDS}
    for model_field, form_field in YEAR_FIELDS.items():
        days = model.get(model_field)
        form[form_field] = None if _blank(days) else round(-float(days) / DAYS_PER_YEAR, 1)
    form[NOT_EMPLOYED] = model.get("DAYS_EMPLOYED") == DAYS_EMPLOYED_SENTINEL
    if form[NOT_EMPLOYED]:
        form["YEARS_EMPLOYED"] = None
    for field, no_score in NO_SCORE.items():
        form[no_score] = form[field] is None
    return form


def label(field: str) -> str:
    """Form label for a model or form field (model day fields map to their years question)."""
    return FIELDS[YEAR_FIELDS.get(field, field)][0]


def friendly_error(message: str) -> str:
    """'EXT_SOURCE_2: must be between 0 and 1 (got 1.3)' -> 'External credit score 2 (0 to 1): must be ...'."""
    field, _, problem = message.partition(": ")
    return f"{label(field)}: {problem}" if field in FIELDS or field in YEAR_FIELDS else message
