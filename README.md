# SSNR ADL Classification

This repo includes anonymized labeled data for activities of daily living in spinal cord injury individuals. There are also introductions to classification pipelines.

## Cloning the Repository

To get started with this project, first, you need to clone the repository to your local machine. Open your terminal and run the following command:

'git clone https://github.com/SCAI-Lab/ssnr_adl_classification.git'

## Setting Up the Virtual Environment

After cloning the repository, navigate into the project directory:

`cd your-repo-name`

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

Make sure to deactivate the virtual environment when you're done working on the project:

`deactivate`



cd ssnr_adl_classification
source venv/bin/activate

pip install -r requirements.txt

python -m scripts.windowing_and_feature_extraction
