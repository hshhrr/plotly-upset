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
df = pd.DataFrame(np.random.randint(0, 2, size=(10_000, 3)), columns=set_list)

fig = plot_upset(
	dataframes=[df],
	legendgroups=["Group X"]
)

fig.show()
```
[example-01](https://raw.githubusercontent.com/hshhrr/plotly-upset/main/img/example-01.png?raw=true)