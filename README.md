
<div align="center">
	<img alt="plotly-upset-icon" style="width:256px;height:256px;" src="https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/plotly-upset.png?raw=true">
</div>


<h1 align="center">
    plotly-upset
</h1>

<p align="center">
    <i>
        UpSet intersection visualization utility for Plolty (Python-only)
    </i>
</p>


<!-- # plotly-upset
UpSet intersection visualization method for Plolty (Python-only) -->

## Installation

```bash
pip install git+https://github.com/hshhrr/plotly-upset.git
```

## Examples

### Basic use

```python
import numpy as np
import pandas as pd
from plotly_upset.plotting import plot_upset

# Dummy Data
set_list = ["Set A", "Set B", "Set C"]
df = pd.DataFrame(
    np.random.randint(0, 2, size=(10_000, len(set_list))), columns=set_list
)

# Plotting
fig = plot_upset(
    dataframes=[df],
    legendgroups=["Group X"],
    marker_size=16,
)

fig.update_layout(
    font_family="Jetbrains Mono",
)

fig.show()
```

![example-01](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-01.png?raw=true)

### Scaling (Sets)

```python
# 5 Sets
set_list = ["Set A", "Set B", "Set C", "Set D", "Set E"]
df = pd.DataFrame(
    np.random.randint(0, 2, size=(10_000, len(set_list))), columns=set_list
)

# Plotting
fig = plot_upset(
    dataframes=[df],
    legendgroups=["Group X"],
    column_widths=[0.2, 0.8],
    horizontal_spacing = 0.075,
    marker_size=10,
)

fig.update_layout(
    width=800,
    font_family="Jetbrains Mono",
)

fig.show()
```

![example-02](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-02.png?raw=true)

### Scaling (Dataframes)

```python
# 3 Dataframes
set_list = ["Set A", "Set B", "Set C"]
df0 = pd.DataFrame(
    np.random.randint(0, 2, size=(10_000, len(set_list))), columns=set_list
)
df1 = pd.DataFrame(
    np.random.randint(0, 2, size=(10_000, len(set_list))), columns=set_list
)
df2 = pd.DataFrame(
    np.random.randint(0, 2, size=(10_000, len(set_list))), columns=set_list
)

# Plotting
fig = plot_upset(
    dataframes=[df0, df1, df2],
    legendgroups=["Group X", "Group Y", "Group Z"],
    marker_size=16,
)

fig.update_layout(
    height=500,
    width=800,
    font_family="Jetbrains Mono",
)

fig.show()
```

![example-03](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-03.png?raw=true)

### Custom Marker Colors

```python
# Dummy Data
set_list = ["Set A", "Set B", "Set C"]
df0 = pd.DataFrame(
    np.random.randint(0, 2, size=(10_000, len(set_list))), columns=set_list
)
df1 = pd.DataFrame(
    np.random.randint(0, 2, size=(10_000, len(set_list))), columns=set_list
)

# Custom Marker Colors
cmc = ["#651FFF", "#00E676"]

# Plotting
fig = plot_upset(
    dataframes=[df0, df1],
    legendgroups=["Group X", "Group Y"],
    marker_colors=cmc,
    marker_size=16,
)

fig.update_layout(
    font_family="Jetbrains Mono",
)

fig.show()
```

![example-04](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-04.png?raw=true)


### Sorting and Removing Zero Values

```python
# Data - Source https://github.com/hms-dbmi/upset-altair-notebook
df = pd.read_csv(
    'https://raw.githubusercontent.com/hms-dbmi/upset-altair-notebook/master/data/covid_symptoms_table.csv',
    usecols=lambda x: x != 'id'
)

# Plotting
fig = plot_upset(
    dataframes=[df],
    legendgroups=["COVID-19 Symptoms"],
    exclude_zeros=True,
    sorted_x="d",
    sorted_y="a",
    column_widths=[0.2, 0.8],
    horizontal_spacing = 0.21,
    marker_size=10,
)

fig.update_layout(
    width=830,
    font_family="Jetbrains Mono",
)

fig.show()
```

![example-05](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-05.png?raw=true)


### Adding Marginal Plots

```python
# Data - Source https://github.com/hms-dbmi/UpSetR/blob/master/inst/extdata/movies.csv
movies = pd.read_csv(
    "https://raw.githubusercontent.com/hms-dbmi/UpSetR/master/inst/extdata/movies.csv",
    sep=';'
)

# Preprocessing - Taking 4 categories with most number of samples
df = movies.drop(columns=['Name', 'ReleaseDate', 'AvgRating', 'Watches'])
x = [(df[cat].sum(), cat) for cat in df]
x = np.array(sorted(x, reverse=True))
x = x.T[1][:4]
df = movies[x]

# Plotting
fig = plot_upset(
    dataframes=[df],
    legendgroups=["Movie Categories - X"],
    exclude_zeros=True,
    sorted_x="d",
    sorted_y="a",
    row_heights=[0.6, 0.4],
    vertical_spacing = 0.,
    horizontal_spacing = 0.15,
    marginal_data=[movies['ReleaseDate'], movies['AvgRating'], movies['Watches']],
    marginal_title=['ReleaseDate', 'AvgRating', 'Watches']
)

fig.update_layout(
    height=650,
    width=830,
    font_family="Jetbrains Mono",
)

fig.show()
```

![example-06](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-06.png?raw=true)


### Adding Marginal Plots - Both Axis

```python
# Plotting
fig = plot_upset(
    dataframes=[df],
    legendgroups=["Movie Categories - X"],
    exclude_zeros=True,
    sorted_x="d",
    sorted_y="a",
    column_widths=[0.3, 0.7],
    vertical_spacing = 0.,
    horizontal_spacing = 0.05,
    marginal_y=True,
    marginal_data=[movies['ReleaseDate'], movies['AvgRating']],
    marginal_title=['ReleaseDate', 'AvgRating'],
)

fig.update_layout(
    height=600,
    width=830,
    font_family="Jetbrains Mono",
)

fig.show()
```

![example-07](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-07.png?raw=true)


## Citation

If you use an UpSet figure in a publication using this library, please cite the [original paper](https://vdl.sci.utah.edu/publications/2014_infovis_upset/).

```bibtex
@article{2014_infovis_upset,
    title = {UpSet: Visualization of Intersecting Sets},
    author = {Alexander Lex and Nils Gehlenborg and Hendrik Strobelt and Romain Vuillemot and Hanspeter Pfister},
    journal = {IEEE Transactions on Visualization and Computer Graphics (InfoVis)},
    doi = {10.1109/TVCG.2014.2346248},
    volume = {20},
    number = {12},
    pages = {1983--1992},
    year = {2014}
}
```