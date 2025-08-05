git clone https://github.com/benedekrozemberczki/karateclub.git
cd karateclub
"%PYTHON%" -m pip install .

cd ..
rmdir "karateclub" /q /s
"%PYTHON%" setup.py install --single-version-externally-managed --record=%TEMP%\record.txt
if errorlevel 1 exit 1
