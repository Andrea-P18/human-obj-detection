YOLOv8 Human Detection
----------------------
Questo codice permette di rilevare persone in webcam live o in video pre-registrati utilizzando YOLOv8 e la libreria supervision per le annotazioni

Requisiti:
-----
Python 3.10 (in quanto La versione Python 3.12 dava problemi con cuda)

Le librerie importate in main.py:
    ultralytics (YOLO)
    supervision
    opencv-python
    torch
    
Eseguire il codice all'interno di un ambiente in cui sono installate tutte le librerie

-----------------------------------------

PER ESEGUIRE IL CODICE

da terminale, posizionarsi nella cartella del progetto e lanciare:

python -m main --mode {cam|video} --webcam-resolution w h --path video_path --dev {cpu|cuda}

-----------------------------------------
Parametri principali:

mode: modalità di esecuzione
    cam -> utilizza la webcam
    video -> analizza un video preregistrato
    
webcam-resolution: risoluzione della webcam (width,heigth)
    utile solo se si utilizza --mode cam
    
path: percorso del video da analizzare
    utile solo se si utilizza --mode video
    parametro per adesso inutile perché non è stata prevista la sua inclusione in quanto il     seguente codice è solo un esempio
    
dev: device su cui eseguire il modello
    cpu -> esecuzione su cpu
    cuda -> esecuzione su gpu (se disponibile)


