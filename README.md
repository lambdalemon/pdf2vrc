# pdf2vrc
## What is this for?
Encode pdfs into textures so that you can bring books with you in VRChat. Designed for VRChat avatars though should work for worlds or Unity projects in general.
## Install
We recommed Anaconda for python package management.
```
conda create -n pdf2vrc python=3.10
conda activate pdf2vrc
pip install -r requirements.txt
```
Also download `msdf-atlas-gen.exe` from https://github.com/Chlumsky/msdf-atlas-gen and place it in this folder
## Usage
```
python pdf2vrc.py yourpdf.pdf
```
For more options
```
python pdf2vrc.py --help
```
If successful, two textures, one `.png` and one `.exr` will be generated and placed in the `out` directory. The script will also print out any necessary info not included in the textures.
## Known Missing Features
### General
- Any color that isn't black
- Transparency
- Raster images
### Text
- Horizontal scaling
### Curve
- Curve fill with holes
- Curve fill with self-intersection
- Line cap & join style (currently defaults to hard cap for a single straight line and round otherwise)
