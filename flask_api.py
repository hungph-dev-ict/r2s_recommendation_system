from flask import Flask, jsonify, make_response
from flask_restful import reqparse, abort, Api, Resource
from RecSys_using_CF_test.RecSys_using_CF import RecSys_CF
import pandas as pd

#load dữ liệu vào
interactions_full_check = pd.read_csv('course_user.csv')

#model
model = RecSys_CF(interactions_full_check)


app = Flask(__name__)
api = Api(app)

class RecommendItemsToUser(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('appId')
        parser.add_argument('userId')
        parser.add_argument('count', type=int)
        self.args = parser.parse_args()
        return self.verify()

    def verify(self):
        list_verify = interactions_full_check[interactions_full_check.app_id == int(self.args["appId"])].user_id.unique()
        if int(self.args["userId"]) in list_verify:
            return self.recommend()
        else:
            return self.not_valid(404)

    def fit(self):
        appId = int(self.args["appId"])
        model.fit(a_id = appId)

    def recommend(self):
        self.fit()
        return jsonify(model.recommend_items_to_user(int(self.args["userId"]), int(self.args["count"])))

    @app.errorhandler(404)
    def not_valid(self, error):
        return make_response(jsonify({'error': 'UserID is not exist !'}), 404)

api.add_resource(RecommendItemsToUser, "/r2s/v1/recommend/itemstouser")

if __name__ == '__main__':
    app.run(debug=True)
