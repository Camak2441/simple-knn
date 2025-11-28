simple-knn
---

It computes the **average distance to the nearest neighbors** for a set of 3D points.

This fork adds a pyproject.toml so that it can be installed without --no-build-isolation. 

Install:
```bash
pip install git+https://github.com/camenduru/simple-knn

# or 
git clone https://github.com/camenduru/simple-knn && cd simple-knn
pip install .
```

Usage:
```python
from simple_knn._C import distCUDA2

# shape: [N, 3]
demopc = torch.from_numpy(np.load("/path")).float().cuda().contiguous() 

# shape: [N]
mean_distances = distCUDA2(demopc)
```
