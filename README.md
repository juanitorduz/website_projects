[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/juanitorduz/website_projects)

[![Updated Badge](https://badges.pufler.dev/updated/juanitorduz/website_projects)](https://github.com/juanitorduz/website_projects)
[![GitHub stars](https://img.shields.io/github/stars/juanitorduz/website_projects.svg)](https://github.com/juanitorduz/website_projects/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/juanitorduz/website_projects.svg?color=blue)](https://github.com/juanitorduz/website_projects/network)

# Website Projects

This repository contains the code of the projects presented in my personal website [https://juanitorduz.github.io](https://juanitorduz.github.io).

- Feel free to share and use the code.

- If you find a mistake or typo please let me know (ideally report as an issue or open a [pull request](https://help.github.com/en/articles/about-pull-requests)).

- Comments and suggestions are always welcome.

---

contact: [juanitorduz@gmail.com](mailto:juanitorduz@gmail.com)

---

### Local Development 
### PyEnv

- Create environment variables in your shell:

```
pipenv install --dev
```

- Activate:

```
pipenv shell
```

## Conda Environment

*Warning:* Support for [pyenv](https://github.com/pyenv/pyenv) is going to be the default in the future. I will keep the [environment.yml](environment.yml) file for reference.

*For Winidows uses please see the [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)*.

- [Create conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

  `conda env create -f environment.yml`

- Activate conda environment:

  `conda activate website_projects`

- Run [Jupyter Notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html):

  `jupyter notebook`

  OR

- Run [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/index.html#):

  `jupyter lab`
