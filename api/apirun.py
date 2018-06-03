from flask import Flask,jsonify,abort,request
from until.getScore import getScore
app = Flask(__name__)





@app.route('/api/get_score', methods=['POST'])
def create_task():
    if not request.json or not ('username' in request.json and 'password' in request.json):
        abort(400)
    user = request.json
    username = user['username']
    password = user['password']


    if username and password:
        data = getScore(username, password)
        return jsonify({'data': data}), 201
    else:

        abort(400)

if __name__ == '__main__':
    app.run(debug=True)
