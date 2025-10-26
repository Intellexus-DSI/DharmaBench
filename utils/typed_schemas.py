"""
Schemas for structured output
"""

###############################################################################
# Imports
from typing import TypedDict, List
from typing_extensions import Annotated

###############################################################################
# TypedDict Schemas

class TagTextPair(TypedDict):
    LABEL: str
    SPAN: str


class Prediction(TypedDict):
    """Prediction"""
    prediction: Annotated[
        List[TagTextPair],
        "Nested list of predicted dictionaries with LABEL and SPAN"
    ]

class Label(TypedDict):
    """Label"""

    label: Annotated[
        str,
        ...,
        "Label for the classification task, e.g., 'positive', 'negative', 'neutral'",
    ]

###############################################################################

TYPED_SCHEMAS = {
    "detection": Prediction,
    "classification": Label,
}
