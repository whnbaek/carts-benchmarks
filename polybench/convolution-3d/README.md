# convolution-3d - 3D Convolution (3×3×3 Kernel)

## Description

3D convolution with 3×3×3 filter kernel. Used in video processing and 3D CNNs.

## Original Source

**Repository**: [Polybench-ACC](https://github.com/cavazos-lab/PolyBench-ACC)
**Original Path**: `OpenMP/stencils/convolution-3d/`
**License**: See Polybench LICENSE file

**Polybench Project**:
- **Official Site**: http://polybench.sourceforge.net/
- **Citation**: Pouchet, L.N., et al. "Polybench: The polyhedral benchmark suite." IMPACT, 2012.

## Algorithm

3×3×3 convolution (27-point stencil):
```
B[i][j][k] = Σ(l=-1 to 1) Σ(m=-1 to 1) Σ(n=-1 to 1)
             weight[l][m][n] * A[i+l][j+m][k+n]
```

## Problem Sizes

- **MINI**: 32×32×32
- **SMALL**: 64×64×64
- **STANDARD**: 256×256×256

## Use in Machine Learning

3D convolution is used in:
- Video processing (spatial + temporal dimensions)
- 3D medical imaging (CT, MRI scans)
- Point cloud processing
- Action recognition in videos

## CARTS Compatibility

- ✅ No global variables
- ✅ Clean parameter passing
- ✅ OpenMP parallel with collapse(2)
- ✅ 27-point stencil pattern

## References

- **Repository**: https://github.com/cavazos-lab/PolyBench-ACC
- **3D CNNs**: Ji, S., et al. "3D Convolutional Neural Networks for Human Action Recognition." ICML, 2010.
- **C3D**: Tran, D., et al. "Learning Spatiotemporal Features with 3D Convolutional Networks." ICCV, 2015.
