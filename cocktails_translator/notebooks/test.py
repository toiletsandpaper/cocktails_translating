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
    """
    Convert a token to a float. The token might be a Unicode fraction or a standard number.
    """
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
# Each unit maps to a tuple with (target_unit, conversion_factor).
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
    "unit": ("whole", 1),
    "splash": ("splash", 1),
    "whole": ("whole", 1),
    "ml (4 drops)": ("ml", 1),
}

# Set of tokens considered as descriptors/modifiers rather than measurements.
# KNOWN_MODIFIERS: set[str] = {"dried", "candied", "fresh", "grated"}
KNOWN_MODIFIERS: set[str] = {"dried", "candied", "fresh", "grated", "zest"}

class Ingredient(NamedTuple):
    quantity: int | float | None
    unit: str
    name: str

def extract_ingredient(ingredient_parsed: list[str]) -> Ingredient:
    """
    Parse a single ingredient given as a list of two strings: 
    the 'quantity token' and the ingredient name.
    
    This function handles composite numeric tokens like "2 1⁄2" and applies unit
    conversion if a known measurement unit is detected.
    
    If an unrecognized token is encountered (i.e. not in KNOWN_UNITS or KNOWN_MODIFIERS),
    an exception is logged.
    """
    qty_token: str = ingredient_parsed[0].strip()
    name_token: str = ingredient_parsed[1].strip()

    tokens: list[str] = qty_token.split()
    
    quantity: float | None = None
    unit: str | None = None
    modifier: str = ""
    
    token_index: int = 0
    num_tokens: int = len(tokens)
    
    if num_tokens == 0:
        logger.exception(f"UNHANDLED empty quantity: {ingredient_parsed}")
        return Ingredient(None, "whole", name_token)
    
    # Process tokens to determine quantity and unit.
    while token_index < num_tokens:
        token: str = tokens[token_index].lower()
        if token in UNICODE_FRACTIONS or token.replace('.', '', 1).isdigit() or '/' in token:
            # Token is a number or fraction.
            if quantity is None:
                quantity = parse_fraction(token)
            else:
                quantity += parse_fraction(token)
        elif token in KNOWN_UNITS:
            # Token is a known unit.
            target_unit, factor = KNOWN_UNITS[token]
            if quantity is not None:
                quantity *= factor
            if target_unit == "cm":
                quantity = 0.5 * round(quantity / 0.5)
            unit = target_unit
        elif token in KNOWN_MODIFIERS:
            # Token is a known modifier.
            modifier = token.capitalize()
            unit = "whole"
        else:
            # Token is unhandled; assume it's part of the unit or descriptor.
            if unit is None:
                unit = token
            else:
                unit += f" {token}"
        token_index += 1

    # Default unit if none was determined.
    if unit is None:
        unit = "whole"

    # Prepend modifier to name if modifier isn't already present.
    if modifier and modifier.lower() not in name_token.lower():
        final_name: str = f"{modifier} {name_token}"
    else:
        final_name = name_token

    # Final rounding for quantity.
    if quantity is not None:
        if float(quantity).is_integer():
            quantity = int(quantity)
        else:
            quantity = round(quantity, 2)

    return Ingredient(quantity, unit, final_name)

def extract_ingredients(raw: str) -> list[Ingredient]:
    """
    Extract ingredients from a raw string representing a list of lists.
    Each inner list should contain two elements: a quantity token and an ingredient name.
    
    The raw data is expected to be a one-line string. 
    Example:
    '[["2 1⁄2 test", "Sugar"], ["2 1⁄2", "Sugar"],["2 dried", "Star anise"], ... ]'
    """
    ingredients_list: list[list[str]] = ast.literal_eval(raw)
    extracted: list[Ingredient] = []
    for ingredient in ingredients_list:
        ingr: Ingredient = extract_ingredient(ingredient)
        extracted.append(ingr)
    return extracted

# ---------- Example Testing ----------
if __name__ == "__main__":
    dataset = pl.read_csv('hf://datasets/erwanlc/cocktails_recipe/train.csv')
    #test_data: str = """[["2 1⁄2 test", "Sugar"], ["2 1⁄2", "Sugar"],["2 dried", "Star anise"],["3 pea", "Wasabi paste"],["7 dried", "Clove"],["2 twist", "Orange (fresh fruit)"],["20 dried", "Raisins"],["1 sprig", "Rosemary sprig"],["1⁄4 spoon", "Freshly grated nutmeg"],["1 cupful", "White caster sugar"]]"""
    ingredients: list[Ingredient] = []
    for row in dataset[:]['ingredients']:
        try:
            ingr: list[Ingredient] = extract_ingredients(row)
            ingredients.extend(ingr)
        except Exception as e:
            logger.exception(f"Failed to parse row: {row}")
            
    unique_units = set(ingredient.unit for ingredient in ingredients)
    print("Unique units:", unique_units)
