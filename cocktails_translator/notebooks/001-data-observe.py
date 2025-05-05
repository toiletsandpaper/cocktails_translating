import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import huggingface_hub
    from typing import NamedTuple

    from cocktails_translator.settings import Settings

    settings = Settings()
    return NamedTuple, mo, settings


@app.cell
def _():
    import polars as pl

    dataset = pl.read_csv('hf://datasets/erwanlc/cocktails_recipe/train.csv')
    #dataset['ingridients'] = dataset.select(pl.col('ingredients').cast(pl.List))
    dataset
    return (dataset,)


@app.cell
def _(NamedTuple, dataset, mo):
    import ast

    class Ingredient(NamedTuple):
        quantity: int | float | None
        unit: str
        name: str

    def extract_ingredient(ingredient_parsed: list[str]) -> Ingredient:
        amount = ingredient_parsed[0]
        name = ingredient_parsed[1]
        quantity, unit = None, None

        if any(['cl' in amount,
                'dash' in amount,
                'whole' in amount,
                'spoon' in amount,
                'scoop' in amount,
                'slice' in amount,
                'drop' in amount,
                'pint' in amount,
                'pinch' in amount,
                'grind' in amount,
        ]) and len(amount.split()) == 2:
            parts = amount.split()
            try:
                if parts[0] == '1⁄2':
                    quantity = 0.5
                elif parts[0] == '3⁄4':
                    quantity = 0.75
                else: 
                    quantity = float(parts[0])
                unit = parts[1]
                if unit == 'drop':
                    quantity //= 4
                    unit = 'ml'
                unit = unit.replace('dash', 'ml')

            except ValueError as e:
                pass
        if 'inch' in amount:
            if amount.split()[0] == '1⁄2':
                quantity = 0.5
            elif amount.split()[0] == '3⁄4':
                quantity = 0.75
            elif amount.split()[0] == '1⁄6':
                quantity = 1 / 6
            elif amount.split()[0] == '1⁄4':
                quantity = 0.25
            else:
                quantity = float(amount.split()[0])
            quantity *= 2.5
            quantity = 0.5 * round(quantity / 0.5)
            unit = 'cm'
        if amount.startswith('1 1⁄2'):
            quantity = 1.5
            unit = amount.split()[2]
        if amount.startswith('2 1⁄2'):
            quantity = 2.5
            unit = amount.split()[2]
        if amount.startswith('1 1⁄4'):
            quantity = 1.25
            unit = amount.split()[2]
        if amount.endswith('fill glass with'):
            if amount.startswith('1⁄2'):
                quantity = 0.5
            if amount.startswith('2⁄3'):
                quantity = 0.66
            unit = 'glass'
        if 'dash' in amount:
            try:
                quantity = float(amount.split()[0])
            except ValueError:
                pass
            unit = 'ml'
        if any([
            'Top up' in amount,
            'Float' in amount,
            'Splash' in amount,
            'fresh' in amount,
            'grated zest' in amount,
        ]):
            quantity = None
            unit = amount.replace('with', '').replace('of', '')
        if 'cube' in amount and 'sugar' in name.lower():
            quantity = 5
            unit = 'gram'
        if unit is None or name is None:
            print(f'UNHANDLED messy ingredient: {ingredient_parsed}')
            print(f'\t{ingredient_parsed[0].split()}')
            return []


        if unit == 'cl':
            quantity = float(quantity) * 10
            unit = 'ml'
        if quantity is not None:
            quantity = int(quantity) if float(quantity) == int(quantity) else float(quantity)
        name = ingredient_parsed[1]
        return Ingredient(quantity, unit, name)

    def extract_ingredients(raw: str) -> list[Ingredient]:
        """Extracts ingredients from a raw list of lists.

        Args:
            raw: A list of lists, where each inner list contains the quantity and name of an ingredient.

        Returns:
            A list of Ingredient objects.
        """
        return [extract_ingredient(el) for el in ast.literal_eval(raw)]

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
    return Ingredient, ast, extract_ingredients


@app.cell
def _(dataset, extract_ingredients):
    from tqdm.auto import tqdm

    all_ingredients = []
    for row in dataset[:]['ingredients']:
        try:
            all_ingredients.append(extract_ingredients(row))
        except TypeError as e:
            print(row)
            print(e)
            break
    return (all_ingredients,)


@app.cell
def _(all_ingredients):
    all_ingredients
    return


@app.cell
def _(Ingredient, all_ingredients):
    def format_ingredients(ingredients: list[Ingredient]) -> str:
        res = "Ingredients:\n"
        for pos, ing in enumerate(ingredients):
            res = res + f"\tIngredient {pos+1}:\n"
            res = res + f"\t\tName: {ing.name}\n"
            res = res + f"\t\tAmount: {ing.quantity} {ing.unit}\n"
        return res
    print(format_ingredients(all_ingredients[0]))
    return


@app.cell
def _():
    raise Exception('НЕ ЗАБУДЬ ПРОВЕРИТЬ ПОТОМ УНИКАЛЬНЫЕ ЮНИТЫ ЕЩЁ РАЗ')
    return


@app.cell
def _(settings):
    from langfuse.openai import openai

    openai.langfuse_public_key = settings.LANGFUSE_PUBLIC_KEY
    openai.langfuse_secret_key = settings.LANGFUSE_SECRET_KEY
    openai.langfuse_enabled = True
    openai.langfuse_host = settings.LANGFUSE_HOST

    openai.api_key = settings.OPENAI_API_KEY
    openai.base_url = settings.OPENAI_API_BASE

    openai.langfuse_auth_check()
    return


app._unparsable_cell(
    r"""
    import langfuse

    langfuse.create_prompt(
        name=\"ingredient-translator\",
        type=\"text\",
        prompt=(
    \"\"\"You are a top-level bartender living in Russia. Your English and Russian languages skills way 
    better than C2 level both. Your task is to translate a cocktail recipe from English to Russian language.
    If you see an imperial measurement - translate it to metric or remove it from recipe if its in parentheses.
    If cocktail name can be directly translated - translate is as \"Transliterated Name (Translated Name)\", if name can not be translated - only use \"Transliterated Name\".

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
    		- Amount: 90.0 ml
    	- Ingredient 3:
    		- Name: Lime juice (freshly squeezed)
    		- Amount: 22.5 ml
    	- Ingredient 4:
    		- Name: White caster sugar
    		- Amount: 15.0 ml

    RUSSIAN RECIPE:
    - Cocktail Name: \"Abacaxi Ricaço\" (Абакаши Риса́су)
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
    		- Name: Lime juice (freshly squeezed)
    		- Amount: 22.5 ml
    	- Ingredient 4:
    		- Name: White caster sugar
    		- Amount: 15.0 ml

    \"\"\"
        )
        labels=[\"production\"],  # directly promote to production
        config={
            \"supported_languages\": [\"en\", \"ru\"],
        },
    )
    """,
    name="_"
)


@app.cell
def _(ast):
    ast.literal_eval("[['1 whole', 'Pineapple (fresh)'], ['9 cl', 'Havana Club 3 Year Old rum'], ['2.25 cl', 'Lime juice (freshly squeezed)'], ['1.5 cl', 'White caster sugar']]")
    return


@app.cell
def _(settings):
    print(settings)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
