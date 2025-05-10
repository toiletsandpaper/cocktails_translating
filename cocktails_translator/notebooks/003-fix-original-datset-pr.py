import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from rich import print
    return (pl,)


@app.cell
def _(pl):
    old_dataset = pl.read_csv('hf://datasets/erwanlc/cocktails_recipe/train.csv').drop_nulls()
    old_dataset
    return (old_dataset,)


@app.cell
def _(old_dataset):
    import ast
    from pydantic import BaseModel

    class Ingredient(BaseModel):
        amount: str
        name: str

    class Recipe(BaseModel):
        title: str
        glass: str
        garnish: str
        recipe: str
        ingredients: list[Ingredient]

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
            ingr: Ingredient = Ingredient(amount=ingredient[0], name=ingredient[1])
            extracted.append(ingr)
        return extracted


    extract_ingredients(old_dataset.item(0, 'ingredients'))
    return Recipe, extract_ingredients


@app.cell
def _(Recipe, extract_ingredients, old_dataset):
    new_dataset_python: list[Recipe] = []

    for row in old_dataset.iter_rows(named=True):
        _ingr = extract_ingredients(row['ingredients'])
        _recipe = Recipe(
            title=row['title'],
            glass=row['glass'],
            garnish=row['garnish'],
            recipe=row['recipe'],
            ingredients=_ingr,
        )
        new_dataset_python.append(_recipe)

    len(new_dataset_python)
    return (new_dataset_python,)


@app.cell
def _(new_dataset_python, pl):
    new_dataset_polars = pl.DataFrame(new_dataset_python)
    new_dataset_polars
    return (new_dataset_polars,)


@app.cell
def _(new_dataset_polars):
    from datasets import Dataset

    new_dataset_hf = Dataset.from_polars(new_dataset_polars)
    new_dataset_hf
    return (Dataset,)


@app.cell
def _():
    # new_dataset_hf.push_to_hub('toiletsandpaper/draft_fix', private=True)
    return


@app.cell
def _(Dataset):
    from datasets import load_dataset

    pushed_dataset = load_dataset("erwanlc/cocktails_recipe", revision="pr/3", split='train')
    Dataset.to_polars(pushed_dataset)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
