# plotly-upset
UpSet intersection visualization method for Plolty (Python-only)

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
    legendgroups=["Group X"]
)

fig.update_layout(
    font_family="Jetbrains Mono",
)

fig.show()
```

![example-01](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-01.png?raw=true)

### Scaling (Sets)

```python
# 4 Sets
set_list = ["Set A", "Set B", "Set C", "Set D"]
df = pd.DataFrame(
    np.random.randint(0, 2, size=(10_000, len(set_list))), columns=set_list
)

# Plotting
fig = plot_upset(
    dataframes=[df],
    legendgroups=["Group X"]
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
    legendgroups=["Group X", "Group Y", "Group Z"]
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
)

fig.update_layout(
    font_family="Jetbrains Mono",
)

fig.show()
```

![example-04](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-04.png?raw=true)