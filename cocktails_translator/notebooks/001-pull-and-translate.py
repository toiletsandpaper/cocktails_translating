import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    from typing import NamedTuple

    import duckdb
    import structlog
    import polars as pl
    import marimo as mo
    import huggingface_hub
    from rich import print

    from cocktails_translator.settings import Settings


    settings = Settings()

    import logging
    import structlog
    from structlog.stdlib import LoggerFactory
    from structlog.processors import JSONRenderer, TimeStamper, StackInfoRenderer, format_exc_info

    # Configure logging to only capture logs from the 'logger' instance
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("./logs/application.log"),
        ],
    )

    # Configure structlog
    structlog.configure(
        processors=[
            TimeStamper(fmt="iso"),
            StackInfoRenderer(),
            format_exc_info,
            JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger("translating")
    logging.getLogger().setLevel(logging.INFO)  # Suppress logs from other loggers

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
    dataset = pl.read_csv('hf://datasets/erwanlc/cocktails_recipe/train.csv').drop_nulls()
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
    print("Unique quantity:", unique_units)
    return


@app.cell
def _(ingredients, print):
    # double test
    zest_names = set(ingredient.name for ingredient in ingredients if 'zest' in ingredient.name)
    print('Names with zest:', zest_names)
    return


@app.cell
def _(ingredients, print):
    unique_quantity = set(ingredient.quantity for ingredient in ingredients)
    print("Unique :", unique_quantity)
    return


@app.cell
def _(Ingredient, ingredients_by_row, print):
    def format_ingredients(ingredients: list[Ingredient]) -> str:
        res = ''
        for pos, ing in enumerate(ingredients):
            res = res + f"\t- Ingredient {pos+1}:\n"
            res = res + f"\t\t- Name: {ing.name}\n"
            res = res + f"\t\t- Amount: {ing.quantity} {ing.unit}\n"
        return res.replace('\t', ' ' * 4)
    print(format_ingredients(ingredients_by_row[0]))
    return (format_ingredients,)


@app.cell
def _(print):
    from pydantic import BaseModel, Field

    class TranslatedIngredient(BaseModel):
        name: str = Field(
            description="Ingredient name translated to Russian. Use the original ingredient name translation (for example, 'Ананас (свежий)')."
        )
        amount: str = Field(
            description="Ingredient amount in Russian using the format 'QUANTITY UNIT' (e.g., '3 мл' or '1 шт.'). Convert imperial measurements if needed."
        )

    class TranslatedRecipe(BaseModel):
        name: str = Field(
            description="Cocktail name in Russian following the format: 'Transliterated Name (Translated Name)' if a direct translation is available; if not, provide only the transliterated name."
        )
        glass: str = Field(
            description="Type of glass to serve the cocktail in Russian."
        )
        garnish: str = Field(
            description="Garnish instructions for the cocktail in Russian."
        )
        recipe: str = Field(
            description="Detailed cocktail recipe instructions in Russian with all measurements converted to metric (except traditional culinary measurements such as 'pinch', which become 'щепотка')."
        )
        ingredients: list[TranslatedIngredient] = Field(
            description="A list of ingredients in Russian. Each ingredient must include its translated name and amount using the specified format."
        )

    print(TranslatedRecipe.model_json_schema())

    return BaseModel, TranslatedRecipe


@app.cell
def _(
    TranslatedRecipe,
    format_ingredients,
    ingredients_by_row,
    print,
    settings,
):
    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key=str(settings.LANGFUSE_PUBLIC_KEY),
        secret_key=str(settings.LANGFUSE_SECRET_KEY),
        host=str(settings.LANGFUSE_HOST),
    )

    prompt_client = langfuse.create_prompt(
        name="ingredient-translator-chat",
        type="chat",
        prompt=[
            {
                "role": "system",
                "content": """
    You are a highly skilled bartender living in Russia with impeccable fluency in both English and Russian (beyond C2 level proficiency). Your task is to translate an English cocktail recipe into Russian, adhering strictly to the instructions below.

    1. Translation Requirements:
       - Translate all text into Russian.
       - When encountering imperial measurements, convert them to metric units. For measurements provided within parentheses, either convert to metric or remove them if redundant.
       - Ensure the output strictly follows the provided structure.

    2. Cocktail Name:
       - Translate the cocktail name to follow the format: "Transliterated Name (Translated Name)" if a direct translation is available.
       - If a direct translation is not possible, output only the transliterated name (avoid empty parentheses) in format: "Transliterated Name".

    3. Glass, Garnish, and Recipe:
       - Translate these elements literally into Russian.
       - Ensure the recipe text uses metric measurements (ml, mm, etc.), except for traditional culinary terms (like "pinch," which should be translated as "щепотка").

    4. Ingredients:
       - Provide a list of ingredients, each with its translated name in Russian and the amount formatted as "QUANTITY UNIT" (e.g., "3 мл" or "1 шт."). Convert any imperial measurements as necessary.

    Below is an UNCHANGED EXAMPLE of the ENGLISH RECIPE and the corresponding expected RUSSIAN output for your reference:

    --------------------------------------------------------------------------------------------------
    EXAMPLE ENGLISH RECIPE:
    - Cocktail Name: Abacaxi Ricaço
    - Glass: Pineapple shell (frozen) glass
    - Garnish: Cut a straw sized hole in the top of the pineapple shell & replace it as a lid.
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

    EXAMPLE RUSSIAN RECIPE (Expected Output):
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
    --------------------------------------------------------------------------------------------------

    Please ensure your output exactly matches the JSON schema provided in the backend configuration, with the fields and structure for cocktail name, glass, garnish, recipe, and ingredients as specified. Do not include any additional explanation or formatting outside the expected structure.
                """
            },
            {
                "role": "user",
                "content": """
    YOU SHOULD TRANSTALE THIS ENGLISH RECIPE:
    - Cocktail Name: {{name}}
    - Glass: {{glass}}
    - Garnish: {{garnish}}
    - Recipe: {{recipe}}
    - Ingredients:
    {{ingredients}}

    TO RUSSIAN LANGUAGE ACCORDING TO YOUR SYSTEM PROMPT IN A JSON SCHEMA.
                """
            },
        ],
        labels=["production"],
        config={
            "supported_languages": ["en", "ru"],
            "output_model": TranslatedRecipe.model_json_schema(),
        }
    )

    print(prompt.compile(
        name='NAME',
        glass="GLASS",
        garnish="GARNISH",
        recipe="RECIPE",
        ingredients=format_ingredients(ingredients_by_row[0]),
    ))

    return (prompt,)


@app.cell
def _(BaseModel, dataset, format_ingredients, ingredients_by_row, print):
    class OriginalRecipe(BaseModel):
        name: str
        glass: str
        garnish: str
        recipe: str
        ingredients: str


    mapped_rows: list[OriginalRecipe] = []
    for i, _row in enumerate(dataset.iter_rows()):
        try:
            _recipe = OriginalRecipe(
                name=_row[0],
                glass=_row[1],
                garnish=_row[2],
                recipe=_row[3],
                ingredients=format_ingredients(
                    ingredients_by_row[i]
                ),
            )
            mapped_rows.append(_recipe)
        except Exception as e:
            print(_row)
            raise Exception(e)
    print(mapped_rows[:2])
    return OriginalRecipe, mapped_rows


@app.cell
async def _(
    OriginalRecipe,
    TranslatedRecipe,
    logger,
    mapped_rows,
    pl,
    prompt,
    settings,
):
    from pydantic import ValidationError
    from langfuse.model import ChatPromptClient
    from langfuse.decorators import observe
    from langfuse.openai import openai
    import asyncio

    openai.langfuse_public_key = str(settings.LANGFUSE_PUBLIC_KEY)
    openai.langfuse_secret_key = str(settings.LANGFUSE_SECRET_KEY)
    openai.langfuse_host = str(settings.LANGFUSE_HOST)

    openai.api_key = str(settings.OPENAI_API_KEY)
    openai.base_url = str(settings.OPENAI_API_BASE)

    if not openai.langfuse_auth_check():
        raise Exception('Something wrong with langfuse connection.')

    @observe(as_type="generation")
    async def translate_recipe_async(prompt_client: ChatPromptClient, original_recipe: OriginalRecipe) -> TranslatedRecipe | None:
        """
        Uses the provided Langfuse prompt_client (TextPromptClient) and an OriginalRecipe instance to generate
        a translated recipe in Russian. Uses langfuse openai client for text completion with structured output
        enforcing the TranslatedRecipe schema. Retries up to 5 times if the structured output fails to parse.

        This is an async version of the `translate_recipe` function. It uses `await` to handle asynchronous
        operations, such as making API calls to the Langfuse OpenAI client.

        Parameters:
            prompt_client: A Langfuse TextPromptClient to use for processing the prompt.
            original_recipe: An instance of OriginalRecipe containing the cocktail recipe to translate.

        Returns:
            An instance of TranslatedRecipe on success, otherwise None if the translation fails after 5 tries.
        """
        client = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=str(settings.OPENAI_API_BASE),
        )
        max_retries = 5
        attempt = 0
        while attempt < max_retries:
            try:
                compiled_prompt = prompt_client.compile(
                    name=original_recipe.name,
                    glass=original_recipe.glass,
                    garnish=original_recipe.garnish,
                    recipe=original_recipe.recipe,
                    ingredients=original_recipe.ingredients,
                )
                response = await client.beta.chat.completions.parse(
                    model=settings.TRANSLATOR_MODEL_NAME,
                    messages=compiled_prompt,
                    response_format=TranslatedRecipe,
                )
                translated_recipe = TranslatedRecipe.model_validate_json(response.choices[0].message.content)
                return translated_recipe

            except (ValidationError, Exception) as e:
                attempt += 1
                logger.warning(f"Attempt {attempt} failed with error: {e}.")
        logger.error(
            f"Translation failed for recipe '{original_recipe.name}' after {max_retries} attempts. Skipping the recipe."
        )
        return None

    async def translate_recipes_in_batch(prompt_client: ChatPromptClient, recipes_batch: list[OriginalRecipe]) -> list[TranslatedRecipe | None]:
        """
        Translates a batch of recipes asynchronously.

        Parameters:
            prompt_client: A Langfuse TextPromptClient to use for processing the prompts.
            recipes_batch: A list of OriginalRecipe instances to translate.

        Returns:
            A list of TranslatedRecipe instances or None for failed translations.
        """
        tasks = [translate_recipe_async(prompt_client, recipe) for recipe in recipes_batch]
        return await asyncio.gather(*tasks)

    @observe()
    async def run_translation_loop_async():
        """
        Iterates over a list of recipes and translates them in batches asynchronously using the `translate_recipes_in_batch` function.
        Maintains the order of recipes during processing and periodically saves the translated recipes to a dataset.

        The function also logs progress and saves the dataset after every 20 successfully translated recipes.

        Returns:
            None
        """
        batch_size = 10
        translated_recipes = []

        for i in range(0, len(mapped_rows), batch_size):
            batch = mapped_rows[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}. Progress: {len(translated_recipes)}/{len(mapped_rows)}")
            results: list[TranslatedRecipe | None] = await translate_recipes_in_batch(prompt_client, batch)
            translated_recipes.extend(filter(None, results))

            logger.info(f"Saving current state of dataset with rows count: {len(translated_recipes)}")
            pl.DataFrame(translated_recipes).write_parquet(f'cocktails_translator/notebooks/datasets/translated_dataset.parquet')


    await run_translation_loop_async()
    return


@app.cell
def _(pl):
    pl.read_parquet(f'cocktails_translator/notebooks/datasets/translated_dataset.parquet')

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
