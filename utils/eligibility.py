"""
Deterministic eligibility engine for Bharat Seva AI.

This module evaluates structured eligibility_rules from schemes.json
against a user's profile.

It does NOT use Gemini or any LLM.
It does NOT parse free-form eligibility text.

If a scheme does not contain any actual structured eligibility
conditions, the engine returns potentially_eligible instead of
claiming that the user is eligible.
"""


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(value):
    """Normalize text for deterministic comparisons."""

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# =========================================================
# CHECK WHETHER REAL RULES EXIST
# =========================================================

def has_actual_rules(rules):
    """
    Determine whether eligibility_rules contains at least
    one meaningful structured condition.

    Values such as None, empty strings, and empty lists
    are treated as missing rules.
    """

    if not isinstance(rules, dict):
        return False

    for value in rules.values():

        if value is None:
            continue

        if value == "":
            continue

        if value == []:
            continue

        return True

    return False


# =========================================================
# ELIGIBILITY EVALUATION
# =========================================================

def evaluate_eligibility(user_profile, scheme):
    """
    Evaluate a user's profile against a scheme's structured
    eligibility rules.

    Returns:

        {
            "status": "likely_eligible"
                      | "potentially_eligible"
                      | "not_eligible",

            "passed": [],
            "failed": [],
            "unknown": [],
            "missing_information": []
        }

    Status meanings:

        likely_eligible
            All available structured conditions are satisfied.

        potentially_eligible
            Some required information is missing OR the scheme
            has no actual structured eligibility rules.

        not_eligible
            At least one known structured condition has failed.
    """

    rules = scheme.get("eligibility_rules")

    # ---------------------------------------------------------
    # NO ACTUAL STRUCTURED RULES
    # ---------------------------------------------------------

    if not has_actual_rules(rules):

        return {
            "status": "potentially_eligible",
            "passed": [],
            "failed": [],
            "unknown": [
                "This scheme does not currently have structured "
                "eligibility rules in the knowledge base."
            ],
            "missing_information": [
                "scheme_specific_eligibility"
            ]
        }

    passed = []
    failed = []
    unknown = []
    missing_information = []

    # =========================================================
    # AGE
    # =========================================================

    user_age = user_profile.get("age")

    min_age = rules.get("min_age")
    max_age = rules.get("max_age")

    if min_age is not None:

        if user_age is None:

            unknown.append(
                f"Age is required to verify the minimum age "
                f"requirement of {min_age}."
            )

            if "age" not in missing_information:
                missing_information.append("age")

        elif user_age < min_age:

            failed.append(
                f"Age is below the minimum required age of {min_age}."
            )

        else:

            passed.append(
                f"Age satisfies the minimum requirement of {min_age}."
            )

    if max_age is not None:

        if user_age is None:

            unknown.append(
                f"Age is required to verify the maximum age "
                f"requirement of {max_age}."
            )

            if "age" not in missing_information:
                missing_information.append("age")

        elif user_age > max_age:

            failed.append(
                f"Age exceeds the maximum allowed age of {max_age}."
            )

        else:

            passed.append(
                f"Age is within the maximum limit of {max_age}."
            )

    # =========================================================
    # ANNUAL INCOME
    # =========================================================

    user_income = user_profile.get("annual_income")

    max_income = rules.get("max_annual_income")

    if max_income is not None:

        if user_income is None:

            unknown.append(
                f"Annual income is required to verify the maximum "
                f"income limit of ₹{max_income:,.0f}."
            )

            if "annual_income" not in missing_information:
                missing_information.append("annual_income")

        elif user_income > max_income:

            failed.append(
                f"Annual income exceeds the maximum limit of "
                f"₹{max_income:,.0f}."
            )

        else:

            passed.append(
                f"Annual income is within the maximum limit of "
                f"₹{max_income:,.0f}."
            )

    # =========================================================
    # OCCUPATION
    # =========================================================

    allowed_occupations = rules.get("occupation") or []

    user_occupation = normalize_text(
        user_profile.get("occupation")
    )

    if allowed_occupations:

        normalized_allowed = {
            normalize_text(value)
            for value in allowed_occupations
        }

        if not user_occupation:

            unknown.append(
                "Occupation is required to verify the "
                "occupation-specific requirement."
            )

            if "occupation" not in missing_information:
                missing_information.append("occupation")

        elif user_occupation in normalized_allowed:

            passed.append(
                "Occupation matches the scheme's specified "
                "occupation requirement."
            )

        else:

            failed.append(
                "Occupation does not match the scheme's "
                "specified occupation requirement."
            )

    # =========================================================
    # GENDER
    # =========================================================

    required_gender = rules.get("gender")

    if required_gender is not None:

        user_gender = normalize_text(
            user_profile.get("gender")
        )

        normalized_required_gender = normalize_text(
            required_gender
        )

        if not user_gender:

            unknown.append(
                "Gender is required to determine this "
                "eligibility condition."
            )

            if "gender" not in missing_information:
                missing_information.append("gender")

        elif user_gender != normalized_required_gender:

            failed.append(
                f"Gender does not match the required gender "
                f"({required_gender})."
            )

        else:

            passed.append(
                "Gender requirement is satisfied."
            )

    # =========================================================
    # AGRICULTURAL LANDHOLDING
    # =========================================================

    requires_landholding = rules.get(
        "requires_landholding"
    )

    if requires_landholding is True:

        has_landholding = user_profile.get(
            "has_landholding"
        )

        if has_landholding is None:

            unknown.append(
                "Agricultural landholding status is required."
            )

            if "has_landholding" not in missing_information:
                missing_information.append(
                    "has_landholding"
                )

        elif has_landholding is True:

            passed.append(
                "Required agricultural landholding condition "
                "is satisfied."
            )

        else:

            failed.append(
                "The required agricultural landholding "
                "condition is not satisfied."
            )

    # =========================================================
    # DISABILITY
    # =========================================================

    requires_disability = rules.get(
        "requires_disability"
    )

    min_disability_percentage = rules.get(
        "min_disability_percentage"
    )

    if requires_disability is True:

        has_disability = user_profile.get(
            "has_disability"
        )

        if has_disability is None:

            unknown.append(
                "Disability status is required."
            )

            if "has_disability" not in missing_information:
                missing_information.append(
                    "has_disability"
                )

        elif has_disability is False:

            failed.append(
                "The scheme requires a disability condition "
                "that is not satisfied."
            )

        else:

            passed.append(
                "Required disability condition is satisfied."
            )

            if min_disability_percentage is not None:

                disability_percentage = user_profile.get(
                    "disability_percentage"
                )

                if disability_percentage is None:

                    unknown.append(
                        f"Disability percentage is required to "
                        f"verify the minimum requirement of "
                        f"{min_disability_percentage}%."
                    )

                    if (
                        "disability_percentage"
                        not in missing_information
                    ):
                        missing_information.append(
                            "disability_percentage"
                        )

                elif (
                    disability_percentage
                    < min_disability_percentage
                ):

                    failed.append(
                        f"Disability percentage is below the "
                        f"required minimum of "
                        f"{min_disability_percentage}%."
                    )

                else:

                    passed.append(
                        f"Disability percentage satisfies the "
                        f"minimum requirement of "
                        f"{min_disability_percentage}%."
                    )

    # =========================================================
    # EXISTING BUSINESS
    # =========================================================

    requires_existing_business = rules.get(
        "requires_existing_business"
    )

    if requires_existing_business is True:

        has_existing_business = user_profile.get(
            "has_existing_business"
        )

        if has_existing_business is None:

            unknown.append(
                "Existing business status is required."
            )

            if "has_existing_business" not in missing_information:
                missing_information.append(
                    "has_existing_business"
                )

        elif has_existing_business is True:

            passed.append(
                "Existing business requirement is satisfied."
            )

        else:

            failed.append(
                "The scheme requires an existing business."
            )

    # =========================================================
    # NO EXISTING HOUSE
    # =========================================================

    requires_no_existing_house = rules.get(
        "requires_no_existing_house"
    )

    if requires_no_existing_house is True:

        owns_house = user_profile.get(
            "owns_house"
        )

        if owns_house is None:

            unknown.append(
                "House ownership status is required."
            )

            if "owns_house" not in missing_information:
                missing_information.append(
                    "owns_house"
                )

        elif owns_house is True:

            failed.append(
                "The user owns a house, while the scheme "
                "requires the beneficiary not to own an "
                "existing house."
            )

        else:

            passed.append(
                "No existing house condition is satisfied."
            )

    # =========================================================
    # FINAL STATUS
    # =========================================================

    if failed:

        status = "not_eligible"

    elif unknown:

        status = "potentially_eligible"

    else:

        status = "likely_eligible"

    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "missing_information": list(
            dict.fromkeys(missing_information)
        )
    }


# =========================================================
# HUMAN-READABLE SUMMARY
# =========================================================

def get_eligibility_summary(result):
    """
    Convert an eligibility result into a short,
    human-readable summary.
    """

    status = result.get("status")

    if status == "likely_eligible":

        return (
            "Likely eligible — all available structured "
            "eligibility conditions are satisfied."
        )

    if status == "potentially_eligible":

        if (
            "scheme_specific_eligibility"
            in result.get("missing_information", [])
        ):

            return (
                "Potentially eligible — the available "
                "knowledge base does not contain enough "
                "structured eligibility rules to confirm "
                "eligibility."
            )

        return (
            "Potentially eligible — some required information "
            "is still needed to determine eligibility."
        )

    if status == "not_eligible":

        return (
            "Not eligible based on one or more known "
            "eligibility conditions."
        )

    return (
        "Eligibility could not be determined."
    )
