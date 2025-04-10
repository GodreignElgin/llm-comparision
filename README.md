# LLM Comparison Dashboard

## Overview
This repository contains an interactive web application for comparing various Large Language Models (LLMs) across different precision types (INT4, INT8, and FP16). The dashboard visualizes performance metrics such as latency and throughput, as well as quality metrics like BLEU score, ROUGE scores, and other text quality measurements.

## Features
- Interactive visualizations for model performance comparison
- Quality metrics analysis across different precision types
- Detailed model cards with key statistics
- Responsive design for desktop and mobile viewing

## Deployment Instructions for GitHub Pages

### Step 1: Create a GitHub Repository
1. Go to [GitHub](https://github.com) and sign in to your account.
2. Click the "+" icon in the top right corner and select "New repository".
3. Name your repository (e.g., "llm-comparison-dashboard").
4. Set the repository to "Public" (required for GitHub Pages).
5. Click "Create repository".

### Step 2: Push the Web App to GitHub
1. Open your terminal/command prompt.
2. Navigate to the `web_app` directory that contains your dashboard files.
3. Initialize a Git repository (if not already done):
   ```
   git init
   ```
4. Add all files to the staging area:
   ```
   git add .
   ```
5. Commit the files:
   ```
   git commit -m "Initial commit of LLM comparison dashboard"
   ```
6. Add your GitHub repository as a remote:
   ```
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
   ```
7. Push the files to GitHub:
   ```
   git push -u origin main
   ```
   (Note: If you're using an older version of Git, you might need to use `master` instead of `main`.)

### Step 3: Deploy to GitHub Pages
1. On GitHub, navigate to your repository.
2. Click on "Settings" at the top of the repository.
3. Scroll down to the "GitHub Pages" section.
4. Under "Source", select "main" (or "master") from the branch dropdown menu.
5. Click "Save".
6. After a few minutes, your site will be available at `https://YOUR-USERNAME.github.io/YOUR-REPOSITORY-NAME/`.

## Local Development
To view the dashboard locally before deployment:
1. Navigate to the `web_app` directory.
2. Start a simple HTTP server:
   - For Python 3: `python -m http.server 8000`
   - For Python 2: `python -m SimpleHTTPServer 8000`
3. Open your browser and go to `http://localhost:8000`

## Data Sources
The data used in this dashboard comes from evaluations of various LLM models with different precision settings (INT4, INT8, FP16).

## License
This project is licensed under the MIT License - see the LICENSE file for details.
