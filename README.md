

# pasos para levantar el terapueta
1. crea tu entorno virtual
    en linux 
    ```bash 
    pyton -m venv venv
    ```
    en windows
    ```bash

      python -m venv .venv
      ``` 

2.  activar el entorno virtual
    en linux
    ```bash
        source venv/bi/activate
    ```
    en windows 
    ```bash 
    source .venv/Scripts/activate
    ```


3. instalar dependecias 
```bash
pip install -r requirements.txt
```

4. instalar TTS ignorando dependencias 
```bash 
pip install --no-deps TTS==0.22.0
````


### finalmente ejecutar el software 

sadaa
nota si estas de acuerdo en que se use una base de datos para  que se gurade tu sesion para fines de analisis ejecutar 

```bash 
python app_mongo.py 
```

si solo quieres probar el terapeuta y no deseas que que se guarden las sesiones 
```bash 
python app_integrated.py
```





