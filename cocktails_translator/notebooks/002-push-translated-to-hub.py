import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from datasets import Dataset, DatasetBuilder, load_dataset, load_dataset_builder
    return (load_dataset,)


@app.cell
def _(load_dataset):
    dataset = load_dataset('parquet', data_files='cocktails_translator/notebooks/datasets/translated_dataset.parquet')
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    # dataset.push_to_hub('toiletsandpaper/cocktails_recipe_ru_small')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
