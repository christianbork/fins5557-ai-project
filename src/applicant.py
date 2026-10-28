"""Step 9 (screening page): translate between the analyst's form and the model's input fields.

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

# Form field -> (label, help text). Every model input appears once, in its form units.
FIELDS = {
    # Loan
    "NAME_CONTRACT_TYPE": ("Loan type", "Cash loan (fixed amount) or revolving loan (credit line)."),
    "AMT_CREDIT": (
        "Loan amount",
        "Amount to be lent. Also the exposure at default (EAD) in the profit calculation.",
    ),
    "AMT_ANNUITY": ("Regular repayment", "Repayment per instalment. Can't be larger than the loan."),
    "AMT_GOODS_PRICE": (
        "Price of goods financed",
        "For consumer loans: the price of what the loan pays for. Leave blank for cash loans with no goods.",
    ),
    # Income and employment
    "AMT_INCOME_TOTAL": ("Annual income", "Total yearly income before tax."),
    "NAME_INCOME_TYPE": ("Income type", "Main source of income."),
    NOT_EMPLOYED: ("Not employed (e.g. pensioner)", "Tick if the applicant has no current job."),
    "YEARS_EMPLOYED": ("Years in current job", "Ignored if 'Not employed' is ticked."),
    "OCCUPATION_TYPE": ("Occupation", "Leave blank if unknown."),
    "ORGANIZATION_TYPE": (
        "Type of employer",
        "'XNA' is Home Credit's code for no employer (e.g. pensioners).",
    ),
    "NAME_EDUCATION_TYPE": ("Education", "Highest level completed."),
    # Credit history
    "EXT_SOURCE_1": (
        "External credit score 1",
        "0 (worst) to 1 (best). Leave blank if this source has no score.",
    ),
    "EXT_SOURCE_2": (
        "External credit score 2",
        "0 (worst) to 1 (best). Leave blank if this source has no score.",
    ),
    "EXT_SOURCE_3": (
        "External credit score 3",
        "0 (worst) to 1 (best). Leave blank if this source has no score.",
    ),
    "AMT_REQ_CREDIT_BUREAU_YEAR": (
        "Credit bureau enquiries (last year)",
        "Number of credit checks in the past 12 months. Leave blank if unknown.",
    ),
    # Household and stability
    "CNT_CHILDREN": ("Children", "Number of dependent children."),
    "CNT_FAM_MEMBERS": ("Household size", "Number of people in the household, including the applicant."),
    "NAME_HOUSING_TYPE": ("Housing", "Current housing situation."),
    "FLAG_OWN_REALTY": ("Owns property", "Y or N."),
    "FLAG_OWN_CAR": ("Owns a car", "Y or N."),
    "OWN_CAR_AGE": ("Car age (years)", "Leave blank if no car."),
    "REGION_RATING_CLIENT_W_CITY": (
        "Region rating",
        "Lender's rating of the applicant's region: 1 (best) to 3.",
    ),
    "YEARS_SINCE_REGISTRATION": (
        "Years since registration details changed",
        "Years since the address was registered.",
    ),
    "YEARS_SINCE_ID": ("Years since ID document was renewed", "Years since the current ID was issued."),
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
    return form


def label(field: str) -> str:
    """Form label for a model or form field (model day fields map to their years question)."""
    return FIELDS[YEAR_FIELDS.get(field, field)][0]


def friendly_error(message: str) -> str:
    """'EXT_SOURCE_2: must be between 0 and 1 (got 1.3)' -> 'External credit score 2: must be ...'."""
    field, _, problem = message.partition(": ")
    return f"{label(field)}: {problem}" if field in FIELDS or field in YEAR_FIELDS else message
