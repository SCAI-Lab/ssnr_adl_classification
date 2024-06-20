# SSNR ADL Classification

This repo includes anonymized labeled data for activities of daily living in spinal cord injury individuals. There are also introductions to classification pipelines.

## Cloning the Repository

To get started with this project, first, you need to clone the repository to your local machine. To do so, first install git on your machine: https://git-scm.com/downloads

Then open your terminal and run the following command:

`git clone https://github.com/SCAI-Lab/ssnr_adl_classification.git`

## Setting Up the Virtual Environment

After cloning the repository, navigate into the project directory:

`cd <directory to the cloned repo>/ssnr_adl_classification`

It's highly recommended to use a virtual environment to manage your project dependencies. Follow these steps to create and activate a virtual environment with Python 3.10.13:

1. **Create a virtual environment:**

    `python3.10 -m venv venv`

2. **Activate the virtual environment:**

    - On macOS/Linux:

        `source venv/bin/activate`

    - On Windows:

        `.\venv\Scripts\activate`

## Installing Required Libraries

Once the virtual environment is activated, install the necessary libraries using `pip` and the `requirements.txt` file:

`pip install -r requirements.txt`

This will ensure that all the dependencies for the project are installed and properly configured.

---

## Associating Virtual Environment with Jupyter Kernel

To use the created virtual environment with Jupyter notebooks, you need to associate it with a Jupyter kernel. Follow these steps:

1. **Install the IPython kernel for the virtual environment:**

    `python -m ipykernel install --user --name=venv`

2. **Start Jupyter Notebook:**

    `jupyter notebook`

3. **Select the Kernel in Jupyter Notebook:**
    - Open the notebook you want to work on.
    - Go to the `Kernel` menu, select `Change kernel`, and choose `venv`.

---

Make sure to deactivate the virtual environment when you're done working on the project:

`deactivate`

## Instructions and tutorials

In the `notebooks` folder, you will find the following Jupyter notebooks with instructions for loading data:

- `tutorial_signal_processing.ipynb`: Instructions for signal processing.
- `tutorial_classification.ipynb`: Instructions for creating a classification pipeline.
- `tutorial_lime.ipynb`: Instructions for Explaining a model using LIME analysis.
