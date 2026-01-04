from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Configuration for model name."""
    model_name: str = "linear_regression"
     
