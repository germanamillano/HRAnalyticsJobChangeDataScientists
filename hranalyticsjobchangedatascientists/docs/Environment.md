# Virtual Environments
Instructor: German Gustavo Amillano Gamez
Email: a01688663@tec.mx

Follow the instructions below to do the Environments.

### Instructions
1. Clone the project `https://github.com/germanamillano/HRAnalyticsJobChangeDataScientists` on your local computer.
2. Create a virtual environment with `Python 3.10.9`
    * Create venv
        ```
        python3.10 -m venv venv
        ```

    * Activate the virtual environment

        ```
        source venv/bin/activate
        ```

3. Install libraries
    Run the following command to install the other libraries.

    ```bash
    pip install -r '/Users/mac/Documents/GitHub/HRAnalyticsJobChangeDataScientists/requirements-310txt'
    ```
    Verify the installation with this command:
    ```bash
    pip freeze
    ```
    Output:

4. Install Pre-commit in Terminal
    ```
    pre-commit install   
    ```
    
3. Open the `hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists.py` notebook and click on `Run All`. 
    > **IMPORTANT!**  
    Do not forget to select the Python 3.10.9 kernel you have already created.

**Congrats, the notebook is running in a virtual environment with Python 3.10!**
