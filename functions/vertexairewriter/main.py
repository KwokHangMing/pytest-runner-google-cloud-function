import os
import vertexai
import functions_framework
from vertexai.language_models import TextGenerationModel
from flask import jsonify


@functions_framework.http
def vertexai_rewriter(request):
    request_json = request.get_json()
    if request_json and "text" in request_json:
        text = request_json["text"]
        vertexai.init(
            project=os.environ.get("GCP_PROJECT"), location=(os.environ.get("REGION"))
        )
        parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
        }
        model = TextGenerationModel.from_pretrained("text-bison")
        response = model.predict(text, **parameters)
        generated_text = response.text
        return jsonify({generated_text}), 200, {"ContentType": "application/json"}
    else:
        return (
            jsonify({"error": "Invalid request"}),
            400,
            {"ContentType": "application/json"},
        )
