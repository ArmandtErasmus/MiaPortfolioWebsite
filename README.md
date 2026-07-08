# Mia's Portfolio Website

### 1. Structure
index.html - landing page, contains links to linkedin, resume, github and projects(.html)
CNAME - contains reference to www record and target mialangenhoven.com
projects - folder with project folders, images folders and projects.html
projects.html - page where project link buttons are hosted
images - folder with project images to act as thumbnails on projects.html page

#### 2. Uploading Projects
Each project must have its own folder within the projects folder (e.g. projects/project_01)

##### The projects must all have the following structure:
1. engine (optional folder containing additional code modules if a modular design approach is prefered instead of having everything in app.py)
2. app.py - this is the main streamlit app executable, it must be called "app" e.g. app.py and is recognised by streamlit when deploying from a repo
3. requirements.txt - this is required since a streamlit app requires the libraries your project is built on to be installed, if they are not installed, streamlit reads the requirements.txt file and downloads them automatically --- this prevents any errors on project launch.

#### 3. requirements.txt
The requirements.txt should look like this:
```python
streamlit>=1.30
numpy>=1.24
pandas>=2.0
plotly>=5.18
anthropic>=0.40
```

#### 4. Deploying New Projects
Copy and paste this div block and edit the relevant fields:

```html
<a class="project-card" href="https://mia-smoothed-bonus.streamlit.app/" target="_blank">
    <img src="images/bonus_fund.png" alt=bonus_fund" />
    <div class="project-title">Smoothed Bonus Fund Analytics Platform</div>
</a>
```

1. Change `href="https://mia-smoothed-bonus.streamlit.app/`, `src="images/bonus_fund.png" alt=bonus_fund"`, `<div class="project-title">Smoothed Bonus Fund Analytics Platform</div>`
2. Deploy to [https://share.streamlit.io/]
