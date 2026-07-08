# Mia's Portfolio Website

### Structure
index.html - landing page, contains links to linkedin, resume, github and projects(.html)
CNAME - contains reference to www record and target mialangenhoven.com
projects - folder with project folders, images folders and projects.html
projects.html - page where project link buttons are hosted
images - folder with project images to act as thumbnails on projects.html page

### Uploading Projects
Each project must have its own folder within the projects folder (e.g. projects/project_01)

The projects must all have the following structure:
1. engine (optional folder containing additional code modules if a modular design approach is prefered instead of having everything in app.py)
2. app.py - this is the main streamlit app executable, it must be called "app" e.g. app.py and is recognised by streamlit when deploying from a repo
3. requirements.txt - this is required since a streamlit app requires the libraries your project is built on to be installed, if they are not installed, streamlit reads the requirements.txt file and downloads them automatically --- this prevents any errors on project launch.

#### requirements.txt
The requirements.txt should look like this:
```python
streamlit>=1.30
numpy>=1.24
pandas>=2.0
plotly>=5.18
anthropic>=0.40
```
