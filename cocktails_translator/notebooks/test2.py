import ast
from typing import NamedTuple

import structlog
import polars as pl

# Setup logging to print exceptions.
logger = structlog.get_logger()

# Mapping for known Unicode fractions.
UNICODE_FRACTIONS: dict[str, float] = {
    "1⁄2": 0.5,
    "3⁄4": 0.75,
    "1⁄4": 0.25,
    "1⁄3": 1/3,
    "2⁄3": 2/3,
    "1⁄6": 1/6,
}

def parse_fraction(token: str) -> float:
    """Convert a token to a float, handling fractions and numeric values."""
    if token in UNICODE_FRACTIONS:
        return UNICODE_FRACTIONS[token]
    try:
        return float(token)
    except ValueError:
        if '/' in token:
            try:
                num, den = token.split('/')
                return float(num) / float(den)
            except Exception:
                logger.exception(f"Failed to parse fraction token: {token}")
        return 0.0

# Dictionary for known measurement units and their conversion factors to metric.
KNOWN_UNITS: dict[str, tuple[str, float]] = {
    "cl": ("ml", 10),
    "dash": ("ml", 1),
    "inch": ("cm", 2.5),
    "spoon": ("ml", 1),
    "scoop": ("ml", 1),
    "slice": ("ml", 1),
    "drop": ("ml", 0.25),
    "pint": ("ml", 473),
    "pinch": ("ml", 0.36),
    "gram": ("gram", 1),
    "cupful": ("ml", 240),
    "cube": ("gram", 5),
    "sprig": ("whole", 1),
    "twist": ("whole", 1),
    "wedge": ("whole", 1),
    "leaf": ("whole", 1),
    "unit": ("whole", 1)
}

# Set of tokens considered as descriptors/modifiers rather than measurements.
KNOWN_MODIFIERS: set[str] = {"dried", "candied", "fresh", "grated"}

class Ingredient(NamedTuple):
    quantity: int | float | None
    unit: str
    name: str

def extract_ingredient(ingredient_parsed: list[str]) -> Ingredient:
    """Parse a single ingredient given as a list of two strings."""
    qty_token, name_token = map(str.strip, ingredient_parsed)
    tokens = qty_token.split()
    
    quantity, unit, modifier = parse_quantity_and_unit(tokens)
    final_name = prepend_modifier_to_name(modifier, name_token)

    return Ingredient(quantity, unit, final_name)

def parse_quantity_and_unit(tokens: list[str]) -> tuple[float | None, str, str]:
    """Parse quantity and unit from tokens."""
    if not tokens:
        logger.exception("UNHANDLED empty quantity")
        return None, "whole", ""

    quantity = parse_fraction(tokens[0])
    unit = "whole"
    modifier = ""

    if len(tokens) > 1 and tokens[1] in UNICODE_FRACTIONS:
        quantity += parse_fraction(tokens[1])
        token_index = 2
    else:
        token_index = 1

    if token_index < len(tokens):
        token = tokens[token_index].lower()
        if token in KNOWN_UNITS:
            target_unit, factor = KNOWN_UNITS[token]
            quantity *= factor
            unit = target_unit
            if target_unit == "cm":
                quantity = 0.5 * round(quantity / 0.5)
        elif token in KNOWN_MODIFIERS:
            modifier = token.capitalize()
            unit = "whole"
        else:
            logger.exception(f"UNHANDLED unit or modifier: '{token}'")
    
    return round_quantity(quantity), unit, modifier

def round_quantity(quantity: float | None) -> float | None:
    """Round the quantity to an integer if it's whole, otherwise to two decimal places."""
    if quantity is not None:
        return int(quantity) if quantity.is_integer() else round(quantity, 2)
    return None

def prepend_modifier_to_name(modifier: str, name: str) -> str:
    """Prepend modifier to name if not already present."""
    if modifier and modifier.lower() not in name.lower():
        return f"{modifier} {name}"
    return name

def extract_ingredients(raw: str) -> list[Ingredient]:
    """Extract ingredients from a raw string representing a list of lists."""
    try:
        ingredients_list: list[list[str]] = ast.literal_eval(raw)
    except Exception as e:
        logger.exception(f"Failed to parse raw ingredient data: {raw}")
        return []

    extracted: list[Ingredient] = []
    for ingredient in ingredients_list:
        if len(ingredient) != 2:
            logger.exception(f"Invalid ingredient format: {ingredient}")
            continue
        ingr: Ingredient = extract_ingredient(ingredient)
        extracted.append(ingr)
    return extracted

# ---------- Example Testing ----------
if __name__ == "__main__":
    dataset = pl.read_csv('hf://datasets/erwanlc/cocktails_recipe/train.csv')
    ingredients: list[Ingredient] = []
    for row in dataset['ingredients']:
        try:
            ingr: list[Ingredient] = extract_ingredients(row)
            ingredients.extend(ingr)
        except Exception as e:
            logger.exception(f"Failed to parse row: {row}")
