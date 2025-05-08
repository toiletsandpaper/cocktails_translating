import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    from typing import NamedTuple\

    import duckdb
    import structlog
    import polars as pl
    import marimo as mo
    import huggingface_hub
    from rich import print

    from cocktails_translator.settings import Settings

    settings = Settings()
    logger = structlog.get_logger()
    return NamedTuple, ast, logger, mo, pl, print, settings


@app.cell
def _(logger):
    UNICODE_FRACTIONS: dict[str, float] = {
        "1⁄2": 0.5,
        "3⁄4": 0.75,
        "1⁄4": 0.25,
        "1⁄3": 1/3,
        "2⁄3": 2/3,
        "1⁄6": 1/6,
    }

    KNOWN_UNITS: dict[str, tuple[str, float]] = {
        "cl": ("ml", 10),
        "dash": ("ml", 1),
        "inch": ("cm", 2.5),
        "spoon": ("ml", 1),
        "scoop": ("ml", 1),
        "slice": ("ml", 1),
        "drop": ("ml", 0.25),
        "pint": ("ml", 500),
        "pinch": ("pinch", 1),
        "gram": ("gram", 1),
        "cupful": ("cupful", 1),
        "cube": ("gram", 5),
        "sprig": ("sprig", 1),
        "twist": ("twist", 1),
        "wedge": ("wedge", 1),
        "leaf": ("leaf", 1),
        "unit": ("whole", 1),
        "splash": ("splash", 1),
        "whole": ("whole", 1),
        "ml (4 drops)": ("ml", 1),
    }

    KNOWN_MODIFIERS: set[str] = {"dried", "candied", "fresh", "grated", "zest"}

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
    return KNOWN_MODIFIERS, KNOWN_UNITS, UNICODE_FRACTIONS, parse_fraction


@app.cell
def _(NamedTuple):
    class Ingredient(NamedTuple):
        quantity: int | float | None
        unit: str
        name: str
    return (Ingredient,)


@app.cell
def _(pl):
    dataset = pl.read_csv('hf://datasets/erwanlc/cocktails_recipe/train.csv')
    #dataset['ingridients'] = dataset.select(pl.col('ingredients').cast(pl.List))
    dataset
    return (dataset,)


@app.cell
def _(
    Ingredient,
    KNOWN_MODIFIERS,
    KNOWN_UNITS,
    UNICODE_FRACTIONS,
    ast,
    dataset,
    logger,
    mo,
    parse_fraction,
):
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
        if modifier.lower() not in name_token.lower():
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

    mo.md(f'''before: 
    ```python
    {dataset.item(0, 'ingredients')}
    ```

    after: 
    ```python
    {extract_ingredients(dataset.item(0, 'ingredients'))}
    ```
    '''
    )
    return (extract_ingredients,)


@app.cell
def _(Ingredient, dataset, extract_ingredients, logger, print):
    ingredients: list[Ingredient] = []
    ingredients_by_row: list[list[Ingredient]] = []
    for row in dataset[:]['ingredients']:
        try:
            ingr: list[Ingredient] = extract_ingredients(row)
            ingredients.extend(ingr)
            ingredients_by_row.append(ingr)
        except Exception as e:
            logger.exception(f"Failed to parse row: {row}")
            break
    print(
        len(ingredients),
        len(ingredients_by_row),
    )
    return ingredients, ingredients_by_row


@app.cell
def _(ingredients, print):
    unique_units = set(ingredient.unit for ingredient in ingredients)
    print("Unique units:", unique_units)
    return


@app.cell
def _(ingredients, print):
    # double test
    zest_names = set(ingredient.name for ingredient in ingredients if 'zest' in ingredient.name)
    print('Names with zest:', zest_names)
    return


@app.cell
def _(Ingredient, ingredients_by_row, print):
    def format_ingredients(ingredients: list[Ingredient]) -> str:
        res = "Ingredients:\n"
        for pos, ing in enumerate(ingredients):
            res = res + f"\tIngredient {pos+1}:\n"
            res = res + f"\t\tName: {ing.name}\n"
            res = res + f"\t\tAmount: {ing.quantity} {ing.unit}\n"
        return res
    print(format_ingredients(ingredients_by_row[0]))
    return


@app.cell
def _(settings):
    from langfuse.openai import openai
    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )

    openai.langfuse_public_key = settings.LANGFUSE_PUBLIC_KEY
    openai.langfuse_secret_key = settings.LANGFUSE_SECRET_KEY
    openai.langfuse_enabled = True
    openai.langfuse_host = settings.LANGFUSE_HOST

    openai.api_key = settings.OPENAI_API_KEY
    openai.base_url = settings.OPENAI_API_BASE

    openai.langfuse_auth_check()
    return (langfuse,)


@app.cell
def _(print):
    from pydantic import BaseModel, Field

    class TranslatedIngredient(BaseModel):
        name: str = Field(
            description="Ingredient name translated to Russian. Use the original ingredient name translation (e.g., 'Ананас (свежий)')."
        )
        amount: str = Field(
            description="Ingredient amount in Russian using the format 'QUANTITY UNIT' (for example, '3 мл' or '1 шт.'). Convert imperial measurements if needed."
        )

    class TranslatedRecipe(BaseModel):
        cocktail_name: str = Field(
            description="Cocktail name in Russian following the format: 'Transliterated Name (Translated Name)' if a direct translation is possible; if not, provide only the transliterated name."
        )
        glass: str = Field(
            description="Type of glass to serve the cocktail in Russian."
        )
        garnish: str = Field(
            description="Garnish instructions for the cocktail in Russian."
        )
        recipe: str = Field(
            description="Detailed cocktail recipe instructions in Russian with all measurements converted to metric (except traditional cooking measurements such as 'pinch' which become 'щепотка')."
        )
        ingredients: list[TranslatedIngredient] = Field(
            description="A list of ingredients in Russian. Each ingredient must include its name and amount in the precise format."
        )

    # Sample structure for testing
    _cocktail = TranslatedRecipe(
        cocktail_name='Абакаши Риса́су ("Богатый Ананас")',
        glass="Половинка ананаса (замороженная, в виде чаши)",
        garnish="Аккуратно вырезать отверстие размером соломинки в верхней части ананасовой чаши и использовать верхнюю часть как крышку.",
        recipe=(
            "Срезать верхушку небольшого ананаса и аккуратно вынуть мякоть, оставив стенки толщиной примерно 12 мм. "
            "Поместить чашу в морозильную камеру для охлаждения. Удалить твердую сердцевину из мякоти ананаса и выбросить; "
            "крупно нарезать оставшуюся мякоть, добавить остальные ингредиенты и взбить в блендере с порцией колотого льда (примерно 340 мл). "
            "Перелить коктейль в ананасовую чашу и подавать со соломинками. (Мякоти одного ананаса хватит как минимум на две чаши.)"
        ),
        ingredients=[
            TranslatedIngredient(name="Ананас (свежий)", amount="1 шт."),
            TranslatedIngredient(name="Ром Havana Club 3 Year Old", amount="90 мл"),
            TranslatedIngredient(name="Сок лайма (свежевыжатый)", amount="22.5 мл"),
            TranslatedIngredient(name="Сахарный песок (мелкий, касторный)", amount="15 мл"),
        ]
    )

    print(_cocktail)
    print(TranslatedRecipe.model_json_schema())

    return (TranslatedRecipe,)


@app.cell
def _(TranslatedRecipe, langfuse):
    prompt = langfuse.create_prompt(
        name="ingredient-translator",
        type="text",
        prompt="""
    You are a highly skilled bartender living in Russia with impeccable fluency in both English and Russian (level beyond C2). Your task is to translate an English cocktail recipe into Russian following these precise instructions:

    1. Translation Requirements:
       - Translate all text into Russian.
       - When possible, translate measurements from imperial to metric. If an imperial measurement is present inside parentheses, either convert it to its metric equivalent or remove it if it is redundant.
       - The translation must strictly follow the provided example format.

    2. Cocktail Name:  
       - If the cocktail name can be directly translated, output as: 
           Transliterated Name (Translated Name)
       - If no direct translation is available, output only the transliterated name (do not add empty parentheses).

    3. Glass, Garnish, and Recipe:  
       - Translate the descriptions literally into Russian.
       - In the recipe text, ensure that all measurements are in metric units (ml, mm, etc.), except for traditional cooking measurements such as "pinch", which may be translated as “щепотка” and similar terms.

    4. Ingredients:  
       - Each ingredient must be output with its name and amount in Russian.
       - The amount should be in the format: “QUANTITY UNIT” (e.g. “3 мл”, “2 шт.”).  
       - Convert or remove imperially provided measurements appropriately.

    Below is the sample ENGLISH RECIPE and its expected RUSSIAN counterpart as your reference:

    ENGLISH RECIPE:
    - Cocktail Name: Abacaxi Ricaço
    - Glass: Pineapple shell (frozen) glass
    - Garnish: Cut a straw sized hole in the top of the pineapple shell & replace it as a lid
    - Recipe: Cut the top off a small pineapple and carefully scoop out the flesh from the base to leave a shell with 12mm (½ inch) thick walls. Place the shell in a freezer to chill. Remove the hard core from the pineapple flesh and discard; roughly chop the remaining flesh, add other ingredients and BLEND with one 12oz scoop of crushed ice. Pour into the pineapple shell and serve with straws. (The flesh of one pineapple blended with the following ingredients will fill at least two shells).
    - Ingredients:
        - Ingredient 1:
            - Name: Pineapple (fresh)
            - Amount: 1 whole
        - Ingredient 2:
            - Name: Havana Club 3 Year Old rum
            - Amount: 90 ml
        - Ingredient 3:
            - Name: Lime juice (freshly squeezed)
            - Amount: 22.5 ml
        - Ingredient 4:
            - Name: White caster sugar
            - Amount: 15 ml

    RUSSIAN RECIPE:
    - Cocktail Name: Абакаши Риса́су ("Богатый Ананас")
    - Glass: Половинка ананаса (замороженная, в виде чаши)
    - Garnish: Аккуратно вырезать отверстие размером соломинки в верхней части ананасовой чаши и использовать верхнюю часть как крышку.
    - Recipe: Срезать верхушку небольшого ананаса и аккуратно вынуть мякоть, оставив стенки толщиной примерно 12 мм. Поместить чашу в морозильную камеру для охлаждения. Удалить твердую сердцевину из мякоти ананаса и выбросить; крупно нарезать оставшуюся мякоть, добавить остальные ингредиенты и взбить в блендере с порцией колотого льда (примерно 340 мл). Перелить коктейль в ананасовую чашу и подавать со соломинками. (Мякоти одного ананаса хватит как минимум на две чаши.)
    - Ingredients:
        - Ingredient 1:
            - Name: Ананас (свежий)
            - Amount: 1 шт.
        - Ingredient 2:
            - Name: Ром Havana Club 3 Year Old
            - Amount: 90 мл
        - Ingredient 3:
            - Name: Сок лайма (свежевыжатый)
            - Amount: 22.5 мл
        - Ingredient 4:
            - Name: Сахарный песок (мелкий, касторный)
            - Amount: 15 мл

    Use the exact structure as shown in the example. Ensure that the output strictly conforms to the fields specified in the JSON schema provided (translated cocktail name, glass, garnish, recipe, and a list of ingredients with each ingredient's name and amount).

    Output must be valid according to the output model.

    """,
        labels=["production"],
        config={
            "supported_languages": ["en", "ru"],
            "output_model": TranslatedRecipe.model_json_schema(),
        },
    )

    return


@app.cell
def _(ast):
    ast.literal_eval("[['1 whole', 'Pineapple (fresh)'], ['9 cl', 'Havana Club 3 Year Old rum'], ['2.25 cl', 'Lime juice (freshly squeezed)'], ['1.5 cl', 'White caster sugar']]")
    return


@app.cell
def _(print, settings):
    print(settings)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
