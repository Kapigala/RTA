from flask import Flask, render_template, request
from Unpickler import Unpickler

app = Flask(__name__)
model = Unpickler(open('perceptron_model.pkl', 'rb')).load()


@app.route('/')
def index():  # put application's code here
    return render_template('index.html')


@app.route('/api/calculate')
def calculate():  # put application's code here
    return model.predict([
        float(request.args.get('sepal-length')),
        float(request.args.get('sepal-width')),
        float(request.args.get('petal-length')),
        float(request.args.get('petal-width'))
    ])


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
