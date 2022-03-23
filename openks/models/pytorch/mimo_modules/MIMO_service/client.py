import requests
import json

text = {"text": 'Histone deacetylase inhibitor valproic acid (VPA) has been used to increase the reprogramming efficiency of induced pluripotent stem cell (iPSC) from somatic cells, yet the specific molecular mechanisms underlying this effect is unknown. Using an in vitro pre-mature senescence model, we found that VPA treatment increased cell proliferation and inhibited apoptosis through the suppression of the p16/p21 pathway.'}
headers = {'Content-Type' : 'application/json'}
req = requests.post('http://localhost:9997/mimo', data=json.dumps(text), headers=headers)

if (req.status_code == 200):
	statements = json.loads(req.text)
	print(statements)
else:
	print(req)
	print("Error! Status code :" + str(req.status_code))

# json.loads(requests.post('http://localhost:9997/mimo', data=json.dumps(text), headers={'Content-Type' : 'application/json'}).text)
