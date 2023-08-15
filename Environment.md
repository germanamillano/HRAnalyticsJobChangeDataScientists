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
    
4. Open the `hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists.py` notebook and click on `Run All`. 
    > **IMPORTANT!**  
    Do not forget to select the Python 3.10.9 kernel you have already created.

    If everything was ok, you should be able to see the last cell with this output:
    ```bash
    Predictions:	 [263527.   331884.02 221119.   ... 105722.   213199.   459125.66]
    ```
**Congrats, the notebook is running in a virtual environment with Python 3.10!**




pre-commit install   

git status  

git commit -m "Prueba PreCommit"      

git push    