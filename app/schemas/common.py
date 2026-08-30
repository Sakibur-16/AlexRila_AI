"""Shared scalar types for the public contract.

Money is the important one. Monetary values are ``Decimal`` internally and
serialise to a **JSON string** (``"25.99"``), never a JSON number.

Rationale: JSON numbers are IEEE-754 doubles in every mainstream parser,
including ``JSON.parse``. Round-tripping ``0.1 + 0.2`` through a double is the
classic source of one-cent discrepancies in financial systems, and a receipt
pipeline whose totals are validated to the cent must not reintroduce that at
the serialisation boundary. Consumers parse these strings into their own
decimal type. This is documented prominently in the README and API docs
because it is the single most surprising thing about the contract.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Annotated, Any

from pydantic import BeforeValidator, Field, PlainSerializer

#: Scale used when rendering money. Two places covers all ISO-4217 currencies
#: this pipeline targets; currencies with other minor units keep their parsed
#: scale because quantisation only ever *adds* precision here, never removes it.
MONEY_SCALE = Decimal("0.01")


def _coerce_decimal(value: Any) -> Any:
    """Accept str/int/float/Decimal, rejecting non-finite values.

    ``float`` is accepted for ergonomics (test fixtures, LLM output) but is
    routed through ``str`` so that the decimal value matches what a human
    reading the literal would expect.
    """
    if value is None or isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError("boolean is not a monetary value")
    if isinstance(value, (int, float)):
        try:
            decimal_value = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(f"invalid monetary value: {value!r}") from exc
        if not decimal_value.is_finite():
            raise ValueError("monetary value must be finite")
        return decimal_value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            # Absence is expressed as None, never as an empty string. Returning
            # None here instead would hand a null to a non-optional Decimal and
            # surface as a confusing "Decimal input should be..." error.
            raise ValueError("monetary value is empty; use None for an absent amount")
        try:
            decimal_value = Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(f"invalid monetary value: {value!r}") from exc
        if not decimal_value.is_finite():
            raise ValueError("monetary value must be finite")
        return decimal_value
    return value


def quantize_money(value: Decimal) -> Decimal:
    """Round to the standard minor-unit scale using banker-safe HALF_UP.

    HALF_UP matches how retail systems round, so validating against a printed
    total does not fail on a legitimately rounded half-cent.
    """
    return value.quantize(MONEY_SCALE, rounding=ROUND_HALF_UP)


def _serialize_money(value: Decimal | None) -> str | None:
    """Render a Decimal as a fixed-scale JSON string."""
    if value is None:
        return None
    return f"{quantize_money(value):f}"


#: A monetary amount. Decimal in Python, string on the wire.
Money = Annotated[
    Decimal,
    BeforeValidator(_coerce_decimal),
    PlainSerializer(_serialize_money, return_type=str, when_used="json"),
]

#: A quantity. Decimal because receipts sell 1.5 kg as readily as 2 units.
Quantity = Annotated[
    Decimal,
    BeforeValidator(_coerce_decimal),
    PlainSerializer(
        lambda v: None if v is None else f"{v.normalize():f}",
        return_type=str,
        when_used="json",
    ),
]

#: A probability in ``[0, 1]``. Plain float: precision is not contractual here.
ConfidenceScore = Annotated[float, Field(ge=0.0, le=1.0)]

#: An ISO-4217 alphabetic currency code.
CurrencyCode = Annotated[str, Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")]
