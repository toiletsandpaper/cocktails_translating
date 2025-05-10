import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from datasets import Dataset, DatasetBuilder, load_dataset, load_dataset_builder, concatenate_datasets
    return Dataset, concatenate_datasets, load_dataset


@app.cell
def _():
    import polars as pl

    pl_original_dataset = pl.read_csv('hf://datasets/erwanlc/cocktails_recipe/train.csv').drop_nulls()
    len(pl_original_dataset)
    return (pl_original_dataset,)


@app.cell
def _(Dataset, pl_original_dataset):
    original_dataset = Dataset.from_polars(pl_original_dataset)
    original_dataset
    return (original_dataset,)


@app.cell
def _(concatenate_datasets, load_dataset, original_dataset):
    dataset_1 = load_dataset('parquet', data_files='cocktails_translator/notebooks/datasets/translated_dataset_0_4099.parquet', split='train')
    dataset_2 = load_dataset('parquet', data_files='cocktails_translator/notebooks/datasets/translated_dataset_4100_end.parquet', split='train')

    dataset = concatenate_datasets([dataset_1, dataset_2], split='train')

    #_dataset_split1 = _dataset.select_columns()

    dataset = dataset.add_column('original_name', original_dataset['title'])
    dataset = dataset.select_columns(['original_name', 'name', 'glass', 'garnish', 'recipe', 'ingredients'])
    dataset = dataset.rename_column('name', 'translated_name')
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    dataset.push_to_hub('toiletsandpaper/cocktails_recipe_ru')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
