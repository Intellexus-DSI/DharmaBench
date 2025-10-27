"""
Schemas for structured output.
"""
###############################################################################
# Imports
from pydantic import BaseModel, Field
from typing import Optional, List

###############################################################################
# Pydantic Schemas
class TagTextPair(BaseModel):
    LABEL: str = Field(..., description="The predicted label for the text span")
    SPAN: str = Field(..., description="The corresponding text span")

class Prediction(BaseModel):
    """Prediction schema"""
    prediction: List[TagTextPair] = Field(
        ..., description="Nested list of predicted dictionaries with LABEL and SPAN"
    )

class Label(BaseModel):
    """Label"""
    label: str = Field(
        description="The predicted label for the classification task"
    )

###############################################################################

PYDANTIC_SCHEMAS = {
    "detection": Prediction,
    "classification": Label,
}
