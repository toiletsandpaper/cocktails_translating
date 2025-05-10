# Translating [erwanlc/cocktails_recipe](https://huggingface.co/datasets/erwanlc/cocktails_recipe)

This project translates the [erwanlc/cocktails_recipe](https://huggingface.co/datasets/erwanlc/cocktails_recipe) dataset into Russian, including converting measurements to metric units and ensuring high-quality translations for cocktail names, ingredients, and instructions.

## Datasets

### Final Dataset: [toiletsandpaper/cocktails_recipe_ru](https://huggingface.co/datasets/toiletsandpaper/cocktails_recipe_ru)

- **Description**: Full translation of the original dataset (except 2 null rows) using hand-written measurement conversions and LLM translations.
- **LLM Used**: gemma-3-27b-it (vLLM, bfloat16)
- **Inference Engine**: GPUStack with vLLM backend
- **Hardware**: 4 H100 80GB GPUs (4 LLM replicas)
- **Size**: 6,954 rows, 1.05 MB
- **Format**: Parquet
- **DOI**: [10.57967/hf/5395](https://doi.org/10.57967/hf/5395)
- **License**: MIT

### Small Dataset: [toiletsandpaper/cocktails_recipe_ru_small](https://huggingface.co/datasets/toiletsandpaper/cocktails_recipe_ru_small)

- **Description**: Translation of the first 1,000 rows of the original dataset using hand-written conversions and LLM translations.
- **LLM Used**: gemma-3-27b-it-qat
- **Inference Engine**: LM Studio
- **Hardware**: 1 RTX 3090 GPU
- **Size**: 1,000 rows, 98 kB
- **Format**: Parquet
- **DOI**: [10.57967/hf/5375](https://doi.org/10.57967/hf/5375)
- **License**: MIT

## Citation

If you use this project, please cite it as follows:

```yaml
@misc{lubenets_ilya_2025,
  author       = {Lubenets Ilya and Gunko Maxim},
  title        = {cocktails_recipe_ru (Revision e078a90)},
  year         = 2025,
  url          = {https://huggingface.co/datasets/toiletsandpaper/cocktails_recipe_ru},
  doi          = {10.57967/hf/5395},
  publisher    = {Hugging Face}
}
