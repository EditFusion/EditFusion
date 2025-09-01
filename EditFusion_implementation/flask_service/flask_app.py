from typing import List
from flask import Flask, request
from flask_cors import CORS
from train_and_infer.infer import get_predicted_result
import json


class Res:
    """
    Response base class
    """

    def __init__(self, msg, isSuccessful=True, data=None):
        self.msg = msg
        self.data = data
        self.isSuccessful = isSuccessful

    def getResponse(self):
        res_json = json.dumps(self, default=lambda obj: obj.__dict__)
        return res_json, {"Content-Type": "application/json"}


app = Flask(__name__)
CORS(app)


@app.route("/es_predict", methods=["GET"])
def es_predict():
    base_str = request.args.get("base", None)
    ours_str = request.args.get("ours", None)
    theirs_str = request.args.get("theirs", None)

    # 都不能为空且都为字符串
    if any(
        map(
            lambda x: x is None or not isinstance(x, str),
            [base_str, ours_str, theirs_str],
        )
    ):
        return Res("base, ours, theirs should be str", False).getResponse()

    def strip_and_split(s: str) -> List[str]:
        return s.strip("\n").split("\n")

    base, ours, theirs = map(strip_and_split, [base_str, ours_str, theirs_str])
    print(f"{base=}")
    print(f"{ours=}")
    print(f"{theirs=}")
    try:
        result = get_predicted_result(base, ours, theirs)
    except Exception as e:
        return Res(str(e), False).getResponse()

    return Res("success", True, result).getResponse()


if __name__ == "__main__":
    app.run(debug=False, port=5002, host="0.0.0.0")
