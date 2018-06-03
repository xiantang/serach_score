import requests
data={
        "username":"16210506042",
        "password":"123456zjd"
    }
import json
header={
    'Content-Type': 'application/json'
}
data=json.dumps(data)
score=requests.post("http://111.231.255.225:5010/api/get_score",data=data,headers=header).text
print(score)


