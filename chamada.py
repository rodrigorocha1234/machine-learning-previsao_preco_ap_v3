import pandas as pd
import requests
import json
url = "http://localhost:8080/invocations"
payload = {
  "dataframe_records": [
    {"Bairro": "Jardim Botânico", "Zona": "Zona Sul", "Quartos": 3, "Banheiros": 2, "Vagas_Garagem": 2, "Metragem": 85.0},
    {"Bairro": "Jardim Botânico", "Zona": "Zona Sul", "Quartos": 2, "Banheiros": 2, "Vagas_Garagem": 1, "Metragem": 65.0},
    {"Bairro": "Centro", "Zona": "Centro", "Quartos": 2, "Banheiros": 1, "Vagas_Garagem": 1, "Metragem": 55.0},
  ]
}

resposta = requests.post(url, json=payload, timeout=30)

print(json.dumps(
    resposta.json(),
    indent=4,
    ensure_ascii=False
))