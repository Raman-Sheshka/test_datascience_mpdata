import numpy as np
from flask import Flask, request, jsonify, render_template
from wtforms import Form, SubmitField, DecimalField, validators
from mpdatanba.ml_logic.ml_workflow import MlModelWorkflow

app = Flask(__name__)


# pre-load the model
classifier = MlModelWorkflow()
classifier.load_model()


class ParamsForm(Form):
    gp = DecimalField('Matchs joués', places=2, validators=[validators.InputRequired()])
    min = DecimalField('Minutes jouées', places=2, validators=[validators.InputRequired()])
    pts = DecimalField('Points marqués', places=2, validators=[validators.InputRequired()])
    fgm = DecimalField('Paniers réussis', places=2, validators=[validators.InputRequired()])
    fga = DecimalField('Paniers tentés', places=2, validators=[validators.InputRequired()])
    fgp = DecimalField('Pourcentage de réussite aux tirs', places=2, validators=[validators.InputRequired()])
    three_p_made = DecimalField('Paniers à trois points réussis', places=2, validators=[validators.InputRequired()])
    three_pa = DecimalField('Paniers à trois points tentés', places=2, validators=[validators.InputRequired()])
    three_p_pca = DecimalField('Pourcentage de réussite aux tirs à trois points', places=2, validators=[validators.InputRequired()])
    ftm = DecimalField('Lancers francs réussis', places=2, validators=[validators.InputRequired()])
    fta = DecimalField('Lancers francs tentés', places=2, validators=[validators.InputRequired()])
    ftp = DecimalField('Pourcentage de réussite aux lancers francs', places=2, validators=[validators.InputRequired()])
    oreb = DecimalField('Rebonds offensifs', places=2, validators=[validators.InputRequired()])
    dreb = DecimalField('Rebonds défensifs', places=2, validators=[validators.InputRequired()])
    reb = DecimalField('Rebonds totaux', places=2, validators=[validators.InputRequired()])
    ast = DecimalField('Passes décisives', places=2, validators=[validators.InputRequired()])
    stl = DecimalField('Interceptions', places=2, validators=[validators.InputRequired()])
    blk = DecimalField('Contres', places=2, validators=[validators.InputRequired()])
    tov = DecimalField('Balles perdues', places=2, validators=[validators.InputRequired()])
    #submit = SubmitField('Predict')

def classify_player(params):
    prediction = classifier.compute_predict(params)
    label = "Yes" if prediction[0] == 1.0 else "No"
    return label, prediction[0]


@app.route('/')
def index():
    form = ParamsForm(request.form)
    return render_template('index.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ParamsForm(request.form)
    if request.method == 'POST' and form.validate():
        input_data = np.array([[float(form.gp.data),
                                float(form.min.data),
                                float(form.pts.data),
                                float(form.fgm.data),
                                float(form.fga.data),
                                float(form.fgp.data),
                                float(form.three_p_made.data),
                                float(form.three_pa.data),
                                float(form.three_p_pca.data),
                                float(form.ftm.data),
                                float(form.fta.data),
                                float(form.ftp.data),
                                float(form.oreb.data),
                                float(form.dreb.data),
                                float(form.reb.data),
                                float(form.ast.data),
                                float(form.stl.data),
                                float(form.blk.data),
                                float(form.tov.data)]
                               ])
        print(input_data)
        label, y = classify_player(input_data)
        return render_template("results.html",
                               prediction = y,
                               label = label
                               )
    return render_template('index.html', form=form)

@app.route("/test", methods=["POST"])
def api_rest():
    data = request.get_json()  # Get JSON data from the request
    features = np.array([float(x) for x in data["features"]]).reshape(1, -1)  # Convert JSON data to float
    label, y = classify_player(features)
    return jsonify({"prediction": y, "label": label})

if __name__ == '__main__':
    app.run(debug=True)
